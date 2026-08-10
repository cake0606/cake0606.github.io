# CUDA 内存

## 内存空间

| 内存空间 | 作用域 | 生命周期 | 典型用途 | 主要特性 |
| --- | --- | --- | --- | --- |
| Register | 单个线程 | 线程执行期间 | 标量临时变量、循环变量、地址计算 | 延迟最低，数量有限；寄存器过多会降低 occupancy |
| Local memory | 单个线程 | 线程执行期间 | 寄存器溢出、大型线程私有数组 | 名称是 local，但物理上通常位于 device memory，延迟高 |
| Shared memory | 单个 block | block 执行期间 | block 内线程协作、数据复用、规约 | 片上内存，需显式声明和同步，可能有 bank conflict |
| Global memory | 整个 device/grid | 显式分配到释放 | 大数组、tensor 数据、kernel 输入输出 | 容量大，延迟高，性能依赖访问合并和缓存命中 |
| Constant memory | 整个 device/grid | 模块加载到释放 | 所有线程读取相同常量 | 只读；warp 内广播访问效率高 |
| Texture/read-only cache | 整个 device/grid | 绑定资源期间 | 只读数据、空间局部性访问 | 适合特定只读访问模式 |

## Global Memory

global memory 是 CUDA kernel 最常见的数据来源和写回位置。对 elementwise kernel，性能通常受 global memory 带宽限制。

warp 内 32 个线程访问连续地址时，硬件可以将多个线程的访问合并成较少的 memory transaction。连续、对齐、同类型访问通常更容易合并。

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    y[idx] = x[idx];
}
```

上面的访问模式中，相邻线程访问相邻元素。若 `x` 和 `y` 是 contiguous tensor，访问模式适合 coalescing。

stride 访问会降低合并效率：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int offset = idx * stride;
if (offset < N) {
    y[offset] = x[offset];
}
```

`stride > 1` 时，相邻线程访问的地址不再连续，memory transaction 数量可能增加。

## Vectorized Load/Store

vectorized load/store 使用更宽的数据类型一次搬运多个标量元素。

```cuda
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
```

常见映射：

| 标量类型 | 向量类型 | 单次搬运元素数 | 搬运宽度 |
| --- | --- | ---: | ---: |
| `float` | `float4` | 4 | 128 bit |
| `half` | `half2` | 2 | 32 bit |
| `half` | `float4` 搬运 | 8 | 128 bit |

约束：

- 被转换的地址需要满足向量类型的对齐要求。`float4` 访问通常要求 16-byte aligned。
- 输入输出应为 contiguous 数据。
- tail 元素需要单独处理，不能让最后一个向量访问越界。
- `reinterpret_cast` 只负责类型视图转换，不会复制数据，也不会修正未对齐地址。

## Shared Memory

shared memory 通过 `__shared__` 声明，作用域是一个 thread block。

```cuda
__global__ void kernel(float* x, float* y) {
    __shared__ float tile[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    tile[tid] = x[idx];
    __syncthreads();

    y[idx] = tile[tid];
}
```

`__syncthreads()` 是 block 内同步点。所有线程需要到达同一个同步点，否则可能产生死锁或未定义行为。

shared memory 按 bank 组织。同一个 warp 内多个线程访问同一个 bank 的不同地址时，会产生 bank conflict。连续 `float` 访问通常能均匀分布到不同 bank；带 stride 的 shared memory 访问需要检查 bank conflict。

## Host 和 Device 传输

常见数据传输 API：

| API | 作用 |
| --- | --- |
| `cudaMemcpy` | 同步拷贝 host/device/device 间数据 |
| `cudaMemcpyAsync` | 异步拷贝，通常配合 stream 使用 |
| `cudaMallocHost` / `cudaFreeHost` | 分配/释放 pinned host memory |
| `cudaHostAlloc` | 分配 pinned host memory，可指定额外 flag |

`cudaMemcpyAsync` 要真正与 kernel 并发执行，通常需要满足：

- 使用非默认 stream 或明确管理 stream。
- host 端内存是 pinned memory。
- 硬件支持 copy engine 与 kernel overlap。
- 拷贝和 kernel 之间没有隐式同步依赖。

## Unified Memory

`cudaMallocManaged` 分配 unified memory。CPU 和 GPU 使用同一个指针访问同一段逻辑内存。

```cuda
float* x = nullptr;
cudaMallocManaged(&x, N * sizeof(float));

init_on_cpu(x, N);
kernel<<<grid, block>>>(x, N);
cudaDeviceSynchronize();
use_on_cpu(x, N);

cudaFree(x);
```

unified memory 由运行时负责迁移。访问发生在不同处理器之间切换时，可能触发 page migration。可使用 `cudaMemPrefetchAsync` 提前迁移：

```cuda
cudaMemPrefetchAsync(x, N * sizeof(float), device_id, stream);
```

## Pitched Memory

二维数组可使用 `cudaMallocPitch` 获取按行对齐的 device memory。

```cuda
float* d_ptr = nullptr;
size_t pitch = 0;
cudaMallocPitch(&d_ptr, &pitch, width * sizeof(float), height);
```

访问第 `row` 行时使用 byte pitch：

```cuda
char* base = reinterpret_cast<char*>(d_ptr);
float* row_ptr = reinterpret_cast<float*>(base + row * pitch);
float value = row_ptr[col];
```

`pitch` 的单位是 byte，不是元素个数。

## 分配方式选择

| 场景 | 推荐 API |
| --- | --- |
| 常规 device tensor buffer | `cudaMalloc` / `cudaFree` |
| 频繁分配释放且绑定 stream 顺序 | `cudaMallocAsync` / `cudaFreeAsync` |
| host/device 异步传输 | `cudaMallocHost` / `cudaHostAlloc` |
| 简化 CPU/GPU 共享指针管理 | `cudaMallocManaged` |
| 二维数组按行对齐 | `cudaMallocPitch` |
