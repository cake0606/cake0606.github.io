# CUDA 内存

## 内存空间

| 内存空间 | 作用域 | 生命周期 | 典型用途 | 主要特性 |
| --- | --- | --- | --- | --- |
| 寄存器（register） | 单个线程 | 线程执行期间 | 标量临时变量、循环变量、地址计算 | 延迟最低，数量有限；寄存器使用过多会降低占用率 |
| 局部内存（local memory） | 单个线程 | 线程执行期间 | 寄存器溢出、大型线程私有数组 | 名称虽然是 local，但物理上通常位于设备内存中，延迟较高 |
| 共享内存（shared memory） | 单个 block | block 执行期间 | block 内线程协作、数据复用、规约 | 片上内存，需显式声明和同步，可能出现 bank conflict |
| 全局内存（global memory） | 整个 device/grid | 显式分配到释放 | 大数组、张量数据、CUDA 内核输入输出 | 容量大，延迟高，性能依赖访问合并和缓存命中 |
| 常量内存（constant memory） | 整个 device/grid | 模块加载到释放 | 所有线程读取相同常量 | 只读；warp 内广播访问效率高 |
| 纹理/只读缓存（texture/read-only cache） | 整个 device/grid | 绑定资源期间 | 只读数据、空间局部性访问 | 适合特定只读访问模式 |

## 全局内存

全局内存是 CUDA 内核最常见的数据来源和写回位置。逐元素内核的性能通常受全局内存带宽限制。

一个 warp 内的线程访问连续地址时，硬件可以将多个线程的访问合并成较少的内存事务。连续、对齐且类型相同的访问通常更容易合并。

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    y[idx] = x[idx];
}
```

上面的访问模式中，相邻线程访问相邻元素。若 `x` 和 `y` 是连续布局的张量，该访问模式适合合并。

跨步访问会降低合并效率：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int offset = idx * stride;
if (offset < N) {
    y[offset] = x[offset];
}
```

`stride > 1` 时，相邻线程访问的地址不再连续，内存事务数量可能增加。

## 向量化加载与存储

向量化加载与存储使用更宽的数据类型，一次搬运多个标量元素。

```cuda
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
```

常见映射：

| 标量类型 | 向量类型 | 单次搬运元素数 | 搬运宽度 |
| --- | --- | ---: | ---: |
| `float` | `float4` | 4 | 128 位 |
| `half` | `half2` | 2 | 32 位 |
| `half` | 以 `float4` 搬运 | 8 | 128 位 |

约束：

- 被转换的地址需要满足向量类型的对齐要求。`float4` 访问要求地址至少按 16 字节对齐。
- 输入和输出应采用连续布局。
- 尾部元素需要单独处理，不能让最后一次向量访问越界。
- `reinterpret_cast` 只负责类型视图转换，不会复制数据，也不会修正未对齐地址。

## 共享内存

共享内存通过 `__shared__` 声明，作用域是一个 thread block。

```cuda
__global__ void copy_via_shared(const float* x, float* y, int n) {
    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    bool valid = idx < n;

    if (valid) {
        tile[tid] = x[idx];
    }
    __syncthreads();

    if (valid) {
        y[idx] = tile[tid];
    }
}
```

启动 `copy_via_shared` 时，动态共享内存大小至少应为 `block.x * sizeof(float)`。`__syncthreads()` 是 block 内同步点；同一 block 中的线程必须一致地到达该同步点，否则行为未定义。

共享内存按 bank 组织。同一个 warp 内多个线程访问同一个 bank 的不同地址时，会产生 bank conflict。连续的 `float` 访问通常能均匀分布到不同 bank；跨步访问共享内存时需要检查 bank conflict。

## Host 与 Device 之间的数据传输

常见数据传输 API：

| API | 作用 |
| --- | --- |
| `cudaMemcpy` | 在 host、device 之间或 device 之间复制数据；同步行为取决于内存类型和拷贝方向 |
| `cudaMemcpyAsync` | 异步拷贝，通常配合 stream 使用 |
| `cudaMallocHost` / `cudaFreeHost` | 分配/释放 pinned host memory |
| `cudaHostAlloc` | 分配 pinned host memory，可指定额外标志 |

`cudaMemcpyAsync` 要真正与 CUDA 内核并发执行，通常需要满足：

- 使用能够与其他工作并发的 stream，并明确管理依赖。
- host 端内存是 pinned memory。
- 硬件支持 copy engine 与 CUDA 内核重叠执行。
- 拷贝和 CUDA 内核之间不存在阻止重叠的依赖或同步。

## 统一内存

`cudaMallocManaged` 分配统一内存。CPU 和 GPU 使用同一个指针访问同一段逻辑内存。

```cuda
float* x = nullptr;
cudaMallocManaged(&x, N * sizeof(float));

init_on_cpu(x, N);
kernel<<<grid, block>>>(x, N);
cudaDeviceSynchronize();
use_on_cpu(x, N);

cudaFree(x);
```

统一内存由 CUDA Runtime 负责迁移。在不同处理器之间切换访问位置时，可能触发页面迁移。可使用 `cudaMemPrefetchAsync` 预取到目标处理器：

```cuda
cudaMemPrefetchAsync(x, N * sizeof(float), device_id, stream);
```

## Pitched Memory

二维数组可使用 `cudaMallocPitch` 获取按行对齐的设备内存。

```cuda
float* d_ptr = nullptr;
size_t pitch = 0;
cudaMallocPitch(&d_ptr, &pitch, width * sizeof(float), height);
```

访问第 `row` 行时，需要按字节计算行地址：

```cuda
char* base = reinterpret_cast<char*>(d_ptr);
float* row_ptr = reinterpret_cast<float*>(base + row * pitch);
float value = row_ptr[col];
```

`pitch` 的单位是字节，不是元素个数。

## 分配方式选择

| 场景 | 推荐 API |
| --- | --- |
| 常规设备端张量缓冲区 | `cudaMalloc` / `cudaFree` |
| 频繁分配、释放且需要服从 stream 顺序 | `cudaMallocAsync` / `cudaFreeAsync` |
| host/device 异步传输 | `cudaMallocHost` / `cudaHostAlloc` |
| 简化 CPU/GPU 共享指针管理 | `cudaMallocManaged` |
| 二维数组按行对齐 | `cudaMallocPitch` |
