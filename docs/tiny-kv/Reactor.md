
 Reactor 概念          项目实现
  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   事件分离器            Linux epoll
  ────────────────────  ───────────────
   Reactor/Dispatcher    EventLoop
  ────────────────────  ───────────────
   fd 事件注册           Channel
  ────────────────────  ───────────────
   连接处理器            TcpConnection
  ────────────────────  ───────────────
   监听器                TcpServer
  ────────────────────  ───────────────
   主动连接器            TcpClient
  ────────────────────  ───────────────
   跨线程唤醒            eventfd
  ────────────────────  ───────────────
   字节缓存              Buffer

## Buffer
Buffer由TcpConnection来持有， 每个TcpConnection持有input和ouput两个Buffer。
input Buffer保存从socket收到、还未被上层完全处理的数据。
output Buffer保存上层调用send后， 未能立即发送的字节。
```cpp
std::vector<char> storage_ = std::vector<char>(4096);
  size_t read_index_ = 0;
  size_t write_index_ = 0;
```
对应的三个区域是：
```text
0                read_index       write_index         capacity
  │                     │                 │                  │
  ▼                     ▼                 ▼                  ▼
  ┌─────────────────────┬─────────────────┬──────────────────┐
  │ 已经消费的空间      │ 当前可读数据    │ 尚未使用的空间   │
  └─────────────────────┴─────────────────┴──────────────────┘
```
## Channel
Channel负责把fd、关注的epoll事件以及事件发生后的回调绑定在一起。
```cpp
EventLoop* loop_; // Channel所属的EventLoop
int fd_;    // 监听的文件描述符
uint32_t events_ = 0;   // 关注的事件， 比如可读、可写等
bool registered_ = false;   // 是否已加入epoll
uint64_t registration_token_ = 0;   // 加入epoll时获得的唯一编号
```
Channel只知道执行回调函数， 回调函数的具体实现由上层负责。
```cpp
  void SetReadCallback(Callback callback) { read_callback_ = std::move(callback); }
  void SetWriteCallback(Callback callback) { write_callback_ = std::move(callback); }
  void SetErrorCallback(Callback callback) { error_callback_ = std::move(callback); }
  void SetCloseCallback(Callback callback) { close_callback_ = std::move(callback); }
```
## Poller
对Linux epoll进行封装， 负责注册、修改、删除fd， 并等待就绪事件。
## EventLoop
Reactor的核心调度器, 循环等待Poller返回事件， 分发给Channel, 执行跨线程任务和定时器。
```cpp
Poller poller_;

int wakeup_fd_ = -1;    // 由其他线程唤醒EventLoop
std::unique_ptr<Channel> wakeup_channel_;   

std::vector<Functor> pending_;
std::vector<Timer> timers_;

std::thread::id owner_thread_;
```
在EventLoop中维护一个pending_队列， 用于存放别的线程投递的任务， 例如Raft线程返回的RPC结果、TcpConnection::send(data)等。
`RunInLoop`函数在调用线程是loop线程时， 立即执行， 如果不是loop线程， 会调用`QueueInLoop`将任务放进队列中。
```text
Run()
进入while循环
阻塞在epoll_wait
处理epoll_wait返回的事件
处理pending_中的任务
处理过期的timer
```

## TcpConnection
持有连接资源
```cpp
int socket_fd_;
  std::unique_ptr<Channel> channel_;
  Buffer input_;
  Buffer output_;
```
通过线程投递保证网络单线程化

