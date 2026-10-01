
Raft中使用日志来记录所有操作， 所有结点都有自己的日志列表来记录所有请求。
在Raft算法中， 机器分为Leader、Follower、Candidate三种角色。所有客户端都与Leader交互。
### 日志
Raft算法中的日志主要包含三个部分：
LogEntry {
    index
    term
    command
}
其中： 
- index： 这条日志在整个日志序列中的位置
- term： 日志由哪个任期的Leader创建
- command： 状态机最终要执行的操作， 例如select, add等
### Leader选举
Leader在任期内会定期发送心跳。定时器超时以后， 认为Leader已死或不存在。
成为Candidate后会向其他节点发送请求投票的请求， 如果收到了半数以上的投票就可以成为Leader， 一定时间内没有获得足够投票时， 就会进行一轮新的选举。
出现网络分区时， 没有Leader的分区会进行选举。由于
> 判断多数派时， 按照整个集群的成员数量来判断
少数节点的分区就算存在有Leader， 写入日志的节点数量不可能超过半数， 因此不可能提交操作。
在投票过程中， 节点会与candidate比较日志的lastindex和lastterm， 如果收到lastterm更大的投票请求， 会更新自己的term。在term相同时， 通过lastindex来比较。
### 日志同步
Leader 为每个 Follower 维护两个变量：
  nextIndex[node]
  matchIndex[node]
含义分别是：
  - nextIndex：下一次应该从哪个 index 开始向该节点发送。
  - matchIndex：已经确认该节点与 Leader 一致到哪个 index。
Leader通过AppendEntries向Followers同步日志，AppendEntries中包含的消息包括:
- prevLogIndex：本次新日志前一条日志的 index 
- prevLogTerm：前一条日志所属的 Term。
假设Follower的日志落后：
Leader的日志为[1, 2, 3, 4, 5], 而Follower的日志为[1, 2]。
Leader尝试发送: 
  - prevLogIndex = 5 
  - prevLogTerm = Leader.log[6].term
Leader向前回退： 5 -> 4 -> 3 -> 2 
此时匹配成功， Leader从index=3开始补发
如果Follower存在冲突日志：
通过上述过程找到一致的位置， Follower删除一致位置后的日志, 追加Leader的日志， 保持与Leader一致。
 



### 日志提交
正常情况下：客户端向Leader发送命令 -> Leader创建并持久化日志 -> Leader通过AppendEntries复制给Followers -> 多数节点持久化后回复成功 -> Leader推进CommitIndex -> 应用日志到状态机 -> 向客户端返回结果 -> Leader通过LeaderCommit通知Followers提交并应用日志

Leader切换后， 未提交日志怎么办：
1. 日志还未同步出去：
一开始Leader创建日志2, 还未同步时就下线，新的Leader上线后， 收到客户端命令时， 不会创建日志3, 也就是不会出现[1, null, 3]的情况， 会创建日志2, 然后进行同步。在原Leader重新上线后， index相同但是term不同，原Leader删除未提交的日志， 接受新的日志。

2. 日志同步给了少数节点
以5节点为例， 其中A（Leader）, B两个节点同步得到了日志[2A], A下线后， B成为了Leader, 客户端发送新的请求， B创建了新的日志[3B], 把日志同步到多数节点后， 进行提交， CommitIndex=3, 旧日志由当前任期间接提交。




