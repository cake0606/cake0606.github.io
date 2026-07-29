#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xfffffff, val, mask);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[WARP_SIZE];
    // 计算warp和lane
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // warp内规约
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) reduce_smem[warp] = sum;

    __syncthreads();
    // 第一个warp计算最终sum
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(b, sum);
}

template<const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a, float* b, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / wWARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) reduce_smem[warp] = sum;
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(b, sum);
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xfffffff, val, mask));
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xfffffff, val_f32, mask);
    }
    return val_f32;
}

template<const int NUM_THREADS = 256 / 2> 
__global__ void block_all_reduce_sum_f16x2_f32_kernel(float* a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + NUM_WARPS - 1) /WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    // 在half精度下进行2个half累加
    half2 reg_a = HALF2(a[idx]);
    half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);

    if (lane == 0) reduce_smem[warp] = sum_f32;
    __syncthreads();
    if (warp == 0) sum = warp_reduce_sum_f16_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(b, sum);
}

template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    const half_z = __float2half(0.0f);

    half sum_f16 = z;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        sum_f16 += (((idx + i) < N) ? pack_a[i] : z);
    }

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);

    if (lane == 0) {
        reduce_smem[warp] = __half2float(sum_f16);
    }
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16
warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xfffffff, val, mask));
    }
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16 * a, float* y, int N) {
     int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  // warp leaders store the data to shared memory.
  // use float to keep sum from each block and reduce
  // with fp32 inter warps.
  if (lane == 0)
    reduce_smem[warp] = sum_bf16;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  __nv_bfloat16 sum =
      (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0)
    sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __bfloat162float(sum));
}











