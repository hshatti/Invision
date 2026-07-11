/*   
   *******************************************************
   * General Matrix Matrix multiplication test in pure C *
   *     Haitham Shatti haitham.shatti@gmail.com         *
   *******************************************************

  <General Matrix Matrix multiplication test in pure C for Tensorium>

  Copyright (C) <2025> <Haitham Shatti> <haitham.shatti at gmail dot com>

  This library is free software; you can redistribute it and/or modify it 
  under the terms of the GNU Library General Public License as published by 
  the Free Software Foundation; either version 2 of the License, 
  or (at your option) any later version.

  This program is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
  See the GNU Library General Public License for more details.

  You should have received a copy of the GNU Library General Public License 
  along with this library; if not, write to the Free Software Foundation, Inc., 
  51 Franklin Street - Fifth Floor, Boston, MA 02110-1335, USA.
  
*/
// #define EXEC
// #define __AVX2__
#ifdef EXEC
#include <chrono>
#include <vector>
#include <malloc.h>
#include <random>
#include <stdio.h>
#endif

#if defined(__AVX2__)
  #include <immintrin.h>
typedef __m256 float8;
  #define VECTOR_WIDTH 8
  #define vset1(a) _mm256_set1_ps(a)
  #define vload(a) _mm256_loadu_ps(a)
  #define vstore(a,b) _mm256_storeu_ps(a,b)
  #ifdef __FMA__
    #define vmadd(a,b,c) _mm256_fmadd_ps(a,b,c)
  #else
    #define vmadd(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
  #endif
  #define vmul(a,b) _mm256_mul_ps(a,b)
  #define vadd(a,b) _mm256_add_ps(a,b)
  #define vdot2(a, b) _mm256_dp_ps(a, b, 0b11110001)
  inline float vsum(const __m256 a) {
    __m128 r  = _mm_add_ps( _mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    _mm256_zeroupper();
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    // const float r = _mm_extract_ps(r, 0);
    return r[0];
  } 

inline float vdot(const float8& a, const float8& b) {
    const __m256 r = vdot2(a, b);
    return _mm_add_ps(_mm256_castps256_ps128(r), _mm256_extractf128_ps(r, 1))[0];
} 


#elif defined(__ARM_NEON)
  #include <arm_neon.h>
  #define VECTOR_WIDTH 8
typedef float32x4_t float4;
typedef struct float4 float8[2];
inline float8 vset1( const float in){
    float8 out;
    out[0] = vdupq_n_f32(in);
    out[1] = vdupq_n_f32(in);
    return out;
}
inline float8 vload(const float* in){
    float8 out;
    out[0] = vld1q_f32(in);
    out[1] = vld1q_f32(in+4);
    return out;
}
inline void vstore(float* out, const float8& in){
    vst1q_f32(out, in[0]);
    vst1q_f32(out+4, in[1]);
}
float8 vmadd(const float8& a, const float8& b, float8& c){
    c[0] = vmlaq_f32(c[0], a[0], b[0]);
    c[1] = vmlaq_f32(c[1], a[1], b[1]);
    return c;
}
inline float8 vmul(const float8& a, const float8& b){
    float8 c;
    c[0] = vmulq_f32(a[0], b[0]);
    c[1] = vmulq_f32(a[1], b[1]);
    return c;
}
inline float8 vadd(const float8& a, const float8& b){
    float8 c;
    c[0] = vaddq_f32(a[0], b[0]);
    c[1] = vaddq_f32(a[1], b[1]);
    return c;
}

inline float vsum (const float8& a)
{
    float32x4_t r = vaddq_f32(a[0], a[1]);
    /* Step 1: add adjacent pairs: [a,b,c,d] -> [a+b, c+d, a+b, c+d] */
    float32x2_t lo = vadd_f32(vget_low_f32(r),   vget_high_f32(r));

    /* Step 2: horizontal add the two 2‑lane vectors: [s1,s2] -> [s1+s2, s1+s2] */
    float32x2_t s = vpadd_f32(lo, lo);   // {lo0+lo1, lo0+lo1}

    /* Step 3: extract the first lane – that is the total sum. */
    return vget_lane_f32(s, 0);
}
#else
typedef struct {float f[8];} float8; 
  #define VECTOR_WIDTH 8
inline float8 vset1(const float in){
    float8 out;
    #pragma omp simd
    for (long i=0; i<8; i++) out.f[i] = in;
    return out;
}
inline float8 vload(const float* in){
    float8 out;
    #pragma omp simd
    for (long i=0; i<8; i++) out.f[i] = in[i];
    return out;
}
inline void vstore(float* out, const float8& in){
    #pragma omp simd
    for (long i=0; i<8; i++) out[i] = in.f[i];
}
inline float8 vmadd(const float8& a, const float8& b, float8& c){
    #pragma omp simd
    for (long i=0; i<8; i++) c.f[i] += a.f[i]*b.f[i];
    return c;
}
inline float8 vmul(const float8& a, const float8& b){
    float8 c;
    #pragma omp simd
    for (long i=0; i<8; i++) c.f[i] = a.f[i]*b.f[i];
    return c;
}
inline float8 vadd(const float8& a, const float8& b){
    float8 c;
    #pragma omp simd
    for (long i=0; i<8; i++) c.f[i] = a.f[i]+b.f[i];
    return c;
}
inline float vsum(const float8& a){
    float r = 0.0f;
    // #pragma omp simd
    for (long i=0; i<8; i++) r += a.f[i];
    return r;
}
#endif


#define MIN(a,b) (((a)<(b))?(a):(b)) 



namespace blas{ 
    

    // #define TILE_N N

    void sgemm_nn(
        const long M, const long N, const long K, const float ALPHA
        , const float* A, const long lda
        , const float* B, const long ldb
        , const float BETA
        , float* C, const long ldc, const long ithread=0, long nthreads=0){
            
        if (nthreads==0) nthreads = M;
        
        if (ithread + nthreads>M) nthreads = M - ithread;
        
        const long TILE_K = 4;
        const long TILE_M = 4;
        const long TILE_N = 64;
        static_assert(TILE_N>=VECTOR_WIDTH && TILE_N%VECTOR_WIDTH==0, "TILE_N must be larger than zero and multiples of VECTOR_WIDTH"); // because of ymm vector size = 8 floats
        nthreads = ithread + nthreads;
        // Scale C by BETA
        #ifdef EXEC
        if (BETA==0.0f) {
            for (long m=ithread ; m<nthreads; m++)
                #pragma omp simd
                for (long n=0 ; n<N; n++)
                    C[m*ldc + n] = 0.0f;
        } else if (BETA!=1.0f) {
            for (long m=ithread ; m<nthreads; m++)
                #pragma omp simd
                for (long n=0; n<N; n++)
                    C[m*ldc + n] *= BETA;
        }
        #endif
        if (M<=0 || N<=0 || K<=0 || ALPHA==0.0f) return;

        const long REM_M = M%TILE_M;
        const long REM_N = N%TILE_N;
        const long REM_K = K%TILE_K;
        const long ALIGNED_M = M - REM_M;
        const long ALIGNED_N = N - REM_N;
        const long ALIGNED_K = K - REM_K;

        // assert(TILE_K<=8); // because of a0..a7
        #ifdef EXEC
        // printf("    - [n,n] expcted thread CPU cache required : %dB\n", (TILE_M*K /*A*/ + TILE_K*N /*B*/ + TILE_M*TILE_N /*C*/)*sizeof(float));
        #endif
        #pragma omp parallel for
	    for (long tile_m=ithread ; tile_m<nthreads; tile_m += TILE_M )  {
            for (long tile_k=0; tile_k<ALIGNED_K; tile_k+=TILE_K) {
                for (long tile_n=0; tile_n<ALIGNED_N ; tile_n+=TILE_N){
                    const float* BB = B + tile_k*ldb + tile_n;
                    for (long m=tile_m ; m<MIN(tile_m+TILE_M, nthreads); m++)  {
                        const float* AA = A + m*lda+tile_k;
                        float8 a[TILE_K];
                        #pragma unroll
                        for (long i=0; i<TILE_K; i++){
                            a[i] = vset1(ALPHA*AA[i]);
                        }
                        float* CC = C + m*ldc + tile_n;
                        for (long n=0; n<TILE_N ; n += VECTOR_WIDTH){ // ymm vector size = 8 floats
                            float8 c_part = vload(CC + n);
                            #pragma unroll
                            for (long i=0; i<TILE_K; i++){
                                c_part = vmadd(a[i], vload(BB + i*ldb + n), c_part);
                            }
                            vstore(CC + n, c_part);
                        }
                    }
                }
                // Handle remaining N
                if (REM_N>0) {
                    const float* BB = B + tile_k*ldb + (N-REM_N);
                    for (long m=tile_m ; m<MIN(tile_m+TILE_M, nthreads); m++)  {
                        const float* AA = A + m*lda+tile_k;
                        float* CC = C + m*ldc + (N-REM_N);
                        for (long i=0; i<TILE_K; i++){
                            const float* BBB = BB + i*ldb;
                            const float a = ALPHA*AA[i];
                            #pragma unroll 
			    for (long n=0; n<REM_N ; n++){
                                CC[n] += a*BBB[n];
                            }
                        }
                    }
                }
            }
            // Handle remaining K
            if (REM_K>0) {
                for (long tile_n=0; tile_n<N-REM_N ; tile_n+=TILE_N){
                    const float* BB = B + (K-REM_K)*ldb + tile_n;
                    for (long m=tile_m ; m<MIN(tile_m+TILE_M, nthreads); m++)  {
                        const float* AA = A + m*lda+(K-REM_K);
                        float* CC = C + m*ldc + tile_n;
                        #pragma unroll
                        for (long n=0; n<TILE_N ; n += VECTOR_WIDTH){ // ymm vector size = 8 floats
                            float8 c_part = vload(CC + n);
                           #pragma unroll
                            for (long i=0; i<REM_K; i++){
                                float8 a_part = vset1(ALPHA*AA[i]);
                                c_part = vmadd(a_part, vload(BB + i*ldb + n), c_part);
                            }
                            vstore(CC + n, c_part);
                        }
                    }
                }
                // Handle remaining N
                if (REM_N>0) {
                    const float* BB = B + (K-REM_K)*ldb + (N-REM_N);
                    for (long m=tile_m ; m<MIN(tile_m+TILE_M, nthreads); m++)  {
                        const float* AA = A + m*lda+(K-REM_K);
                        float* CC = C + m*ldc + (N-REM_N);
                        #pragma unroll
                        for (long i=0; i<REM_K; i++){
                            const float* BBB = BB + i*ldb;
                            const float a= ALPHA*AA[i];
                            #pragma unroll
                            for (long n=0; n<REM_N ; n++){
                                CC[n] += a*BBB[n];
                            }
                        }
                    }
                }
            }
        }
    }

    void sgemm_nt(
        const long M, const long N, const long K, const float ALPHA
        , const float* A, const long lda
        , const float* B, const long ldb
        , const float BETA
        , float* C, const long ldc, const long ithread=0, long nthreads=0){
            
            
            if (nthreads==0) nthreads = M;
            if (ithread + nthreads>M) nthreads = M - ithread;
            nthreads = ithread + nthreads;
            
            
            const long TILE_K = 128;
            const long TILE_M = 4;
            const long TILE_N = 4;
            static_assert(TILE_K>=VECTOR_WIDTH && TILE_K%VECTOR_WIDTH==0, "TILE_K must be larger than zero and multiples of VECTOR_WIDTH");
            // Scale C by BETA
            #ifdef EXEC
            if (BETA==0.0f) {
                for (long m=ithread ; m<nthreads; m++)
                #pragma omp simd
                for (long n=0 ; n<N; n++)
                C[m*ldc + n] = 0.0f;
            } else if (BETA!=1.0f) {
                for (long m=ithread ; m<nthreads; m++)
                #pragma omp simd
                for (long n=0; n<N; n++)
                C[m*ldc + n] *= BETA;
            }
            #endif
            
            if (M<=0 || N<=0 || K<=0 || ALPHA==0.0f) return;
            
            const long REM_M = M%TILE_M;
            const long REM_N = N%TILE_N;
            const long REM_K = K%TILE_K;
            const long ALIGNED_M = M - REM_M;
            const long ALIGNED_N = N - REM_N;
            const long ALIGNED_K = K - REM_K;
            const long REM_VEC = K % VECTOR_WIDTH;
            const long ALIGNED_VEC = K - REM_VEC;
            
            
            #ifdef EXEC
            // printf("    - [n.t] expcted thread CPU cache required : %dB\n", (TILE_M*K /*A*/ + TILE_N*K /*B*/ + TILE_M*TILE_N /*C*/)*sizeof(float));
            #endif
            #pragma omp parallel for
            for (long tile_m=ithread ; tile_m<nthreads; tile_m+=TILE_M)  {
                const float* tile_a = A + tile_m*lda;
                float* tile_c       = C + tile_m*ldc;
                for (long tile_n=0; tile_n<ALIGNED_N ; tile_n+=TILE_N){
                    for (long m=tile_m ; m<MIN(tile_m + TILE_M, nthreads); m++){
                        const float* AA = A + m*lda;
                        for (long tile_k=0; tile_k<ALIGNED_K; tile_k += TILE_K) {
                            float* CC = C + m*ldc;
                            for (long n=tile_n;n<tile_n+TILE_N; n++){
                                const float* BB = B + n*ldb; 
                                float8 sum8 = vset1(0.0f);
				#pragma unroll
                                for(long k=tile_k; k<tile_k+TILE_K; k += VECTOR_WIDTH){
                                    sum8 = vmadd(vload(AA + k), vload(BB + k), sum8);
                                }
                                float sum = vsum(sum8);
                                if (ALIGNED_K == tile_k+TILE_K){
                                    // for(long k=ALIGNED_K; k<ALIGNED_VEC; k += VECTOR_WIDTH){
                                    //     sum8 = vmadd(vload(AA + k), vload(BB + k), sum8);
                                    // }    
                                    for (long k=ALIGNED_K ; k<K; k++){
                                        sum += AA[k]*BB[k];
                                    }                                   
                                }    
                                // float sum = vsum(sum8); 
                                // for (long k=ALIGNED_VEC ; k<K; k++){
                                //     sum += AA[k]*BB[k];
                                // }                                   
                                CC[n] += ALPHA*sum;
                            }
                            if (ALIGNED_N == tile_n+TILE_N){
                                for (long n=ALIGNED_N; n<N; n++){
                                    const float* BB = B + n*ldb; 
                                    float8 sum8 = vset1(0.0f);
                                    for(long k=tile_k; k<tile_k+TILE_K; k += VECTOR_WIDTH){
                                        sum8 = vmadd(vload(AA + k), vload(BB + k), sum8);
                                    }
                                    float sum = vsum(sum8);
                                    if (ALIGNED_K == tile_k+TILE_K){
                                        // for(long k=ALIGNED_K; k<ALIGNED_VEC; k += VECTOR_WIDTH){
                                        //     sum8 = vmadd(vload(AA + k), vload(BB + k), sum8);
                                        // }    
                                        for (long k=ALIGNED_K ; k<K; k++){
                                            sum += AA[k]*BB[k];
                                        }                                   
                                    }    
                                    // float sum = vsum(sum8); 
                                    // for (long k=ALIGNED_VEC ; k<K; k++){
                                    //     sum += AA[k]*BB[k];
                                    // }                                   
                                    CC[n] += ALPHA*sum;
                                }
                            }    
                        }
                    }
                    
                    // end of K Tiling
                }
            }
    }



#ifdef EXEC


#endif
void sgemm_nn_naive(
    const long M, const long N, const long K, const float ALPHA
    , const float* A, const long lda
    , const float* B, const long ldb
    , const float BETA
    , float* C, const long ldc, const int ithread, int nthreads){
#ifdef EXEC
    if (BETA==0.0f) {
      for (long m=ithread ; m<nthreads; m++)  // M
          #pragma omp simd
          for (long n=0; n<N ; n++)
              C[m*ldc + n] = 0.0f;
    } else if (BETA!=1.0f) {
      for (long m=ithread ; m<nthreads; m++)  // M
          #pragma omp simd
          for (long n=0; n<N ; n++)
              C[m*ldc + n] *= BETA;
    }
#endif

    #pragma omp parallel for
    for (long m=ithread ; m<nthreads; m++)  {   // M
        float* CC = C + m*ldc;
        for (long k=0; k<K; k++) {
            float A_PART = ALPHA*A[m*lda + k];
	    const float* BB = B + k*ldb;		
            #pragma omp simd
            for (long n=0; n<N ; n++){
                CC[n] += A_PART*BB[n];
            }
        }
    }
}
    void sgemm_nt_naive(
        const long M, const long N, const long K, const float ALPHA
        , const float* A, const long lda
        , const float* B, const long ldb
        , const float BETA
        , float* C, const long ldc, const int ithread, int nthreads){
#ifdef EXEC
        if (BETA==0.0f) {
            for (long m=ithread ; m<nthreads; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] = 0.0f;
        } else if (BETA!=1.0f) {
            for (long m=ithread ; m<nthreads; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] *= BETA;
        }
#endif
//        #pragma omp parallel for
        for (long m=ithread ; m<nthreads; m++)  {   // M
            float* CC = C + m*ldc;
            const float* AA = A + m*lda;
            #pragma unroll
            for (long n=0; n<N; n++) {
                const float* BB = B + n*ldb;
                float sum=0;
                #pragma omp simd
                for (long k=0; k<K;k++)
                    sum += AA[k] * BB[k];
                CC[n] += ALPHA*sum;
            }
        }
    }

    void sgemm_tn_naive(
        const long M, const long N, const long K, const float ALPHA
        , const float* A, const long lda
        , const float* B, const long ldb
        , const float BETA
        , float* C, const long ldc, const int ithread, int nthreads){
#ifdef EXEC
        if (BETA==0.0f) {
            for (long m=ithread ; m<nthreads; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] = 0.0f;
        } else if (BETA!=1.0f) {
            for (long m=ithread ; m<nthreads; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] *= BETA;
        }
#endif

        #pragma omp parallel for
        for (long m=ithread ; m<nthreads; m++) {   // M
            float* CC = C + m*ldc;
            for (long k=0; k<K;k++) {
                float A_PART = ALPHA*A[k*lda + m];
                const float* BB = B + k*ldb;
                #pragma omp simd
                for (long n=0; n<N; n++) {
                    CC[n] += A_PART * BB[n];
                }
            }
        }
    }

}

//typedef void (*prn_t)(const char* str, const size_t a, const size_t b, const size_t c,const size_t d,const size_t e,const size_t f,const size_t g,const size_t h, const size_t i, const size_t j, const size_t k);
#ifndef EXEC
 typedef void (*prn_t)(const char* str, const ...);
 prn_t printf = NULL;
#endif

typedef enum CBLAS_ORDER {CblasRowMajor = 101, CblasColMajor = 102} CBLAS_LAYOUT;
typedef enum _CBLAS_TRANSPOSE {CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114} CBLAS_TRANSPOSE; 

//extern "C" __declspec(dllimport) void prn(const char* str, ...);

extern "C" {

#ifndef EXEC
__declspec(dllexport) void set_prn(const prn_t p){ printf = p;}
#endif

__declspec(dllexport) void cpp_sgemm(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
           const int M, const int N, const int K,
           const float alpha, const float *A, const int lda,
           const float *B, const int ldb, const float beta,
           float *C, const int ldc, const int ithread, const int nthreads) {
    if (transA==CblasNoTrans && transB==CblasNoTrans){
        //printf("[nn] M:%d N:%d K:%d ALPHA:%f lda:%d ldb:%d BETA:%f ldc:%d\n", M, N, K, alpha, lda, ldb, beta, ldc);
        blas::sgemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ithread, nthreads);
    } else if (transA==CblasNoTrans && transB==CblasTrans){
        //printf("[nt] M:%d N:%d K:%d ALPHA:%f lda:%d ldb:%d BETA:%f ldc:%d\n", M, N, K, alpha, lda, ldb, beta, ldc);
        blas::sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ithread, nthreads);
    } else if (transA==CblasTrans && transB==CblasNoTrans){    
        //printf("[tn] M:%d N:%d K:%d ALPHA:%f lda:%d ldb:%d BETA:%f ldc:%d\n", M, N, K, alpha, lda, ldb, beta, ldc);
        blas::sgemm_tn_naive(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ithread, nthreads);
    }
	//else perror("not implemented!");
    //prn("done\n");
}

/*
__declspec(dllexport) void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
           const int M, const int N, const int K,
           const float alpha, const float *A, const int lda,
           const float *B, const int ldb, const float beta,
           float *C, const int ldc) {
        if (beta==0.0f) {
            for (long m=0 ; m<M; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] = 0.0f;
        } else if (beta!=1.0f) {
            for (long m=0 ; m<M; m++)  // M
                #pragma omp simd
                for (long n=0; n<N ; n++)
                    C[m*ldc + n] *= beta;
        }
   cpp_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 0, M);

}
*/
} // for C

#ifdef EXEC
#define MKL_INT int


extern "C" {
    __declspec(dllimport)
void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const
            MKL_INT K, const float alpha, const float* A, const MKL_INT lda, const float* B, const
            MKL_INT ldb, const float beta, const float* C, const MKL_INT ldc);
}

void printarray(float* arr, long rows, long cols, long ldim) {
    for (long i=0; i<rows; i++) {
        for (long j=0; j<cols; j++) {
            printf("%.0f ", arr[i*ldim + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_means_and_vars(const long N, float* A) {
    double mean = A[0];
    double max =A[0]; double min = A[0];
    for (long i=1; i<N; i++){
        if (A[i]>max) max = A[i];
        if (A[i]<min) min = A[i]; 
       mean += A[i];
    }
    mean /= N;
    
    double var = 0.0;
    for (long i=0; i<N; i++) 
        var += pow(A[i]-mean, 2);
    if (N>1)
      var /= (N-1);
    printf("mean: %.2f var: %.2f min: %.2f max: %.2f\n", mean, var, min, max);
}

int main(int argc, const char** argv){

    using namespace std;

    #define M 512
    #define N 1024
    #define K 2560

    vector<float> A(M*K, .0f);
    vector<float> B(K*N, .0f);
    vector<float> C(M*N, .0f);
    vector<float> C_ref(M*N, .0f);
    for (long i=0; i<M*K; i++) A[i] = (float)(2+rand()%7);

    for (long i=0; i<K*N; i++) B[i] = (float)(2+rand()%13);

    for (long i=0; i<M*N; i++) C[i] = 0.0f;
    for (long i=0; i<M*N; i++) C_ref[i] = 0.0f;

    // printf("possible CPU allocated cache memory : %dB\n",((blas::TILE_M*K)+(blas::TILE_K*N)+(blas::TILE_M*N))*sizeof(float));
    #define REPEAT 1000

    auto begin = chrono::steady_clock::now();
    // auto end = chrono::steady_clock::now();
    // auto time_naive = chrono::duration_cast<chrono::microseconds>(end - begin).count();

#ifdef MKL
    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
        //blas::sgemm_nn_naive(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
    auto end = chrono::steady_clock::now();
    auto time_naive = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C_ref [n, n]:\n");
    print_means_and_vars(M*N, C_ref.data());
    printf("Time  MKL: %ld us @%f GFLOPs\n\n", time_naive, (float)(2*K-1)*M*N/1e3/(time_naive/REPEAT));

    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A.data(), K, B.data(), K, 0.0f, C_ref.data(), N);
        //blas::sgemm_nn_naive(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
    end = chrono::steady_clock::now();
    time_naive = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C_ref [n, t]:\n");
    print_means_and_vars(M*N, C_ref.data());
    printf("Time  MKL: %ld us @%f GFLOPs\n\n", time_naive, (float)(2*K-1)*M*N/1e3/(time_naive/REPEAT));


    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, A.data(), M, B.data(), N, 0.0f, C_ref.data(), N);
        //blas::sgemm_nn_naive(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
    end = chrono::steady_clock::now();
    time_naive = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C_ref [t, n]:\n");
    print_means_and_vars(M*N, C_ref.data());
    printf("Time  MKL: %ld us @%f GFLOPs\n\n", time_naive, (float)(2*K-1)*M*N/1e3/(time_naive/REPEAT));
#endif

    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        blas::sgemm_nn(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
//        blas::sgemm_nn_naive(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
    #ifdef MKL
    end = chrono::steady_clock::now();
    #else
    auto end = chrono::steady_clock::now();
    #endif

    auto time_blas = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C [n, n]:\n");
    print_means_and_vars(M*N, C.data());
    printf("Time blas: %ld us @%f GFLOPs\n\n", time_blas, (float)(2*K-1)*M*N/1e3/(time_blas/REPEAT));


    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        // blas::sgemm_nt_naive(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
        blas::sgemm_nt(M, N, K, 1.0f, A.data(), K, B.data(), K, 0.0f, C.data(), N);
    end = chrono::steady_clock::now();
    time_blas = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C [n, t] :\n"); 
    print_means_and_vars(M*N, C.data());
    printf("Time blas: %ld us @%f GFLOPs\n\n", time_blas, (float)(2*K-1)*M*N/1e3/(time_blas/REPEAT));


    begin = chrono::steady_clock::now();
    for (long i=0; i<REPEAT; i++)
        blas::sgemm_tn_naive(M, N, K, 1.0f, A.data(), M, B.data(), N, 0.0f, C.data(), N, 0, M - 1);
    end = chrono::steady_clock::now();
    time_blas = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    printf("C [t, n]:\n");
    print_means_and_vars(M*N, C.data());
    printf("Time blas: %ld us @%f GFLOPs\n\ndone\n", time_blas, (float)(2*K-1)*M*N/1e3/(time_blas/REPEAT));


    // printf("A:\n");
    // print_means_and_vars(M*K, A.data());
    // printf("B:\n");
    // print_means_and_vars(K*N, B.data());
    getchar();
    return 0;
    
}
#endif
