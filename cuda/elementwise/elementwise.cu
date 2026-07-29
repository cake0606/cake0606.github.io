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



// blockIdx: block编号， blockDim: 每个block内有多少线程， threadIdx:block内第几个线程
__global__ void elementwise_add_f32_kernel(float* a, float*b, float* c, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		c[idx] = a[idx] * b[idx]
	}
}

__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int N) {
	int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	if ((idx + 3) < N) {
		float4 reg_a = FLOAT4(a[idx]);
		float4 reg_b = FLOAT4(b[idx]);
		float4 reg_c;
		reg_c.x = reg_a.x + reg_b.x;
		reg_c.y = reg_a.y + reg_b.y;
		reg_c.z = reg_a.z + reg_b.z;
		reg_c.w = reg_a.w + reg_b.w;
		FLOAT4(c[idx]) = reg_c;
	}
	else if (idx < N) {
		for (int i = 0; (idx + i) < N; i++) {
			c[idx + i] = a[idx + i] + b[idx + i];
		}
	}
}

__global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		c[idx] = __hadd(a[idx], b[idx]);
	}
}

__global__ void elementwise_add_f16x2_kernel(half* a, half* b, half* c, int N) {
	int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	if ((idx + 1) < N) {
		half2 reg_a = HALF2(a[idx]);
		half2 reg_b = HALF2(b[idx]);
		half2 reg_c;
		reg_c = __hadd2(reg_a, reg_b);
		HALF2(c[idx]) = reg_c;
	} else if (idx < N) {
		c[idx] = __hadd(a[idx], b[idx]);
	}
}

__global__ void elementwise_add_f16x8_kernel(half* a, half* b, half* c, int N) {
	int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
	if ((idx + 7) < N) {
		half2 reg_a_0 = HALF2(a[idx + 0]);
		half2 reg_a_1 = HALF2(a[idx + 2]);
		half2 reg_a_2 = HALF2(a[idx + 4]);
		half2 reg_a_3 = HALF2(a[idx + 6]);

		half2 reg_b_0 = HALF2(b[idx + 0]);
		half2 reg_b_1 = HALF2(b[idx + 2]);
		half2 reg_b_2 = HALF2(b[idx + 4]);
		half2 reg_b_3 = HALF2(b[idx + 6]);

		half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
		reg_c_0 = __hadd2(reg_a_0, reg_b_0);
		reg_c_1 = __hadd2(reg_a_1, reg_b_1);
		reg_c_2 = __hadd2(reg_a_2, reg_b_2);
		reg_c_3 = __hadd2(reg_a_3, reg_b_3);
		HALF2(c[idx + 0]) = reg_c_0;
		HALF2(c[idx + 2]) = reg_c_1;
		HALF2(c[idx + 4]) = reg_c_2;
		HALF2(c[idx + 6]) = reg_c_3;
	} else if (idx < N) {
		for (int i = 0; (idx + i) < N; i++) {
			c[idx + i] = __hadd(a[idx + i], b[idx + i]);
		}
	}
}

__global__ void elementwise_add_f16x8_pack_kernel(half* a, half* b, half* c, int N) {
	int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
	if ((idx + 7) < N) {
		// 8 * 16bites
		half pack_a[8], pack_b[8], pack_c[8];
		// LDST128BITS会把数据转换成float4, 一次性load 128 bits
		LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
		LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

#pragma unroll
		for (int i = 0; i < 8; i+=2) {
			HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
		}
		// 作为float4来一次性store
		LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
	} else if (idx < N) {
		for (int i = 0; (idx + i) < N; i++) {
			c[idx + i] = __hadd(a[idx + i], b[idx + i]);
		}
	}
}
