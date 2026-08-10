# Elementwise

## 基本约束

本文中的 kernel 假设输入输出满足：

- tensor 位于 CUDA device memory。
- tensor 是 contiguous layout。
- 输入输出元素数量一致。
- vectorized load/store 使用的地址满足对应向量类型的 alignment 要求。

常用类型转换宏：

```cu
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
```

`FLOAT4(x[i])` 将 `&x[i]` 解释为 `float4*` 后解引用。该写法不会创建新对象，只改变当前访问使用的类型视图。

## FP32 Add

每个线程处理一个 `float` 元素。

```cu
__global__ void elementwise_add_f32_kernel(
    const float* a,
    const float* b,
    float* c,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

PyTorch C++ extension 侧启动 kernel：

```cu
void elementwise_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int N = a.numel();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    elementwise_add_f32_kernel<<<grid, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N);
}
```

`data_ptr<float>()` 返回 tensor 底层数据地址，并按 `float*` 类型解释。

## FP32x4 Add

每个线程处理 4 个连续 `float` 元素。

```cu
__global__ void elementwise_add_f32x4_kernel(
    float* a,
    float* b,
    float* c,
    int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 3 < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        FLOAT4(c[idx]) = reg_c;
    } else if (idx < N) {
        for (int i = 0; idx + i < N; ++i) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}
```

启动配置按 4 个元素为一组计算：

```cu
int pack_size = 4;
int packs = (N + pack_size - 1) / pack_size;

dim3 block(256);
dim3 grid((packs + block.x - 1) / block.x);
```

`float4` 路径要求 `a + idx`、`b + idx`、`c + idx` 满足 16-byte alignment。最后不足 4 个元素时走 scalar tail。

## FP16 Add

每个线程处理一个 `half` 元素。

```cu
#include <cuda_fp16.h>

__global__ void elementwise_add_f16_kernel(
    const half* a,
    const half* b,
    half* c,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}
```

## FP16x2 Add

`half2` 包含两个 `half`，`__hadd2` 对两个 lane 分别执行 half 加法。

```cu
__global__ void elementwise_add_f16x2_kernel(
    half* a,
    half* b,
    half* c,
    int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 1 < N) {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c = __hadd2(reg_a, reg_b);

        HALF2(c[idx]) = reg_c;
    } else if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}
```

启动配置按 2 个元素为一组计算：

```cu
int pack_size = 2;
int packs = (N + pack_size - 1) / pack_size;

dim3 block(256);
dim3 grid((packs + block.x - 1) / block.x);
```

## FP16x8 Add

每个线程处理 8 个 `half` 元素。计算使用 4 个 `half2`。

```cu
__global__ void elementwise_add_f16x8_kernel(
    half* a,
    half* b,
    half* c,
    int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 7 < N) {
        half2 reg_a_0 = HALF2(a[idx + 0]);
        half2 reg_a_1 = HALF2(a[idx + 2]);
        half2 reg_a_2 = HALF2(a[idx + 4]);
        half2 reg_a_3 = HALF2(a[idx + 6]);

        half2 reg_b_0 = HALF2(b[idx + 0]);
        half2 reg_b_1 = HALF2(b[idx + 2]);
        half2 reg_b_2 = HALF2(b[idx + 4]);
        half2 reg_b_3 = HALF2(b[idx + 6]);

        HALF2(c[idx + 0]) = __hadd2(reg_a_0, reg_b_0);
        HALF2(c[idx + 2]) = __hadd2(reg_a_1, reg_b_1);
        HALF2(c[idx + 4]) = __hadd2(reg_a_2, reg_b_2);
        HALF2(c[idx + 6]) = __hadd2(reg_a_3, reg_b_3);
    } else if (idx < N) {
        for (int i = 0; idx + i < N; ++i) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}
```

该版本减少线程数量。每个线程使用更多寄存器，并执行多组 `half2` 运算。

## FP16x8 Pack Add

pack 版本使用 128-bit load/store 搬运 8 个 `half`。

```cu
__global__ void elementwise_add_f16x8_pack_kernel(
    half* a,
    half* b,
    half* c,
    int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 7 < N) {
        __align__(16) half pack_a[8];
        __align__(16) half pack_b[8];
        __align__(16) half pack_c[8];

        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }

        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    } else if (idx < N) {
        for (int i = 0; idx + i < N; ++i) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}
```

`LDST128BITS` 使用 `float4` 作为 128-bit 搬运类型。计算仍然通过 `half2` 完成。

## ReLU

ReLU 定义：

```text
y = max(x, 0)
```

FP16x8 pack 版本：

```cu
__global__ void relu_f16x8_pack_kernel(const half* x, half* y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 7 < N) {
        __align__(16) half pack_x[8];
        __align__(16) half pack_y[8];
        half2 zero2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));

        LDST128BITS(pack_x[0]) = LDST128BITS_CONST(x[idx]);

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), zero2);
        }

        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else if (idx < N) {
        half zero = __float2half(0.0f);
        for (int i = 0; idx + i < N; ++i) {
            y[idx + i] = __hmax(x[idx + i], zero);
        }
    }
}
```

边界检查必须在 128-bit load 之前执行。否则当 `N` 不是 8 的倍数时，最后一个 pack 可能读取到数组范围之外。

## Elementwise Kernel 关注项

| 项目 | 说明 |
| --- | --- |
| contiguous | vectorized load/store 依赖连续存储 |
| alignment | `float4`/128-bit load 需要满足地址对齐要求 |
| tail | `N` 不是 pack size 整数倍时需要 scalar fallback |
| register | 单线程处理元素越多，通常寄存器使用越多 |
| occupancy | block size、寄存器、shared memory 都会影响 occupancy |
| memory bandwidth | 简单 elementwise 运算通常主要受 global memory bandwidth 影响 |
