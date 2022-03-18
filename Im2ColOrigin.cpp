#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <ctime>

#if _WIN64
void gemm_nn(int M, int N, int K, float ALPHA,
    float* A, int lda,
    float* B, int ldb,
    float beta,
    float* C, int ldc)
{
    int i, j, k;
//#pragma omp parallel for
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}
#else
extern"C"
{
    #include<cblas.h>
}
#endif


using namespace std;

// 原始的Im2Col
void im2col_cpu(float** src, const int &inHeight, const int &intWidth, const int &kHeight, 
                const int &kWidth, float* srcIm2col){
    const int outHeight = inHeight - kHeight + 1;
    const int outWidth = intWidth - kWidth + 1;
    int cnt = 0;
    for(int i = 0; i < kHeight; i++){
        for(int j = 0; j < kWidth; j++){
            int id = i * kWidth + j;
            int ii = i;
            for(int x = 0; x < outHeight; x++){
                int jj = j;
                for(int y = 0; y < outWidth; y++){
                    srcIm2col[cnt] = src[ii][jj];
                    jj += 1;
                    cnt++;
                }
                ii += 1;
            }
        }
    }
}


constexpr int mec_im2col_orig_kernel_num = 64;
constexpr int mec_im2col_orig_kernel_h = 7;
constexpr int mec_im2col_orig_kernel_w = 7;
constexpr int mec_im2col_orig_in_height = 224;
constexpr int mec_im2col_orig_in_hidth = 224;

// Currently supports ONLY single input channel

float measure_im2col_time(int repeat_count, int out_channels = mec_im2col_orig_kernel_num,
    int kernel_h = mec_im2col_orig_kernel_h, int kernel_w = mec_im2col_orig_kernel_w,
    int in_height = mec_im2col_orig_in_height, int in_width = mec_im2col_orig_in_hidth)
{
    // 构造输入矩阵
    float** src = new float* [in_height];
    for (int i = 0; i < in_height; i++) {
        src[i] = new float[in_width];
        for (int j = 0; j < in_width; j++) {
            src[i][j] = 0.1;
        }
    }
    // 构造kernel矩阵
    float*** kernel = new float**[out_channels];
    for (int i = 0; i < out_channels; i++) {
        kernel[i] = new float* [kernel_h];
        for (int j = 0; j < kernel_h; j++) {
            kernel[i][j] = new float[kernel_w];
            for (int k = 0; k < kernel_w; k++) {
                kernel[i][j][k] = 0.2;
            }
        }
    }


    // 对kernel进行Im2col
    float* kernel2col = new float[out_channels * kernel_h * kernel_w];
    int cnt = 0;
    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < kernel_h; j++) {
            for (int k = 0; k < kernel_w; k++) {
                kernel2col[cnt++] = kernel[i][j][k];
            }
        }
    }

    // 开始计时
    // Changed using: https://www.cplusplus.com/reference/ctime/time/
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    auto start = std::clock();

    for (int repeat_ix = 0; repeat_ix < repeat_count; ++repeat_ix)
    {
        // 对输入矩阵Im2col
        int outHeight = in_height - kernel_h + 1;
        int outWidth = in_width - kernel_w + 1;
        float* srcIm2col = new float[kernel_w * kernel_h * outWidth * outHeight];
        im2col_cpu(src, in_height, in_width, kernel_h, kernel_w, srcIm2col);

        // 使用Blas库实现矩阵乘法
        float* output = new float[out_channels * outHeight * outWidth];

#if _WIN64
        gemm_nn(
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
#endif
            /* M: */        out_channels,
            /* N: */        outHeight * outWidth,
            /* K: */        kernel_w * kernel_h,
            /* Alpha: */    1,
            /* *A: */       kernel2col,
            /* lda: */      kernel_h * kernel_w,
            /* *B: */       srcIm2col,
            /* ldb: */      outHeight * outWidth,
            /* beta: */     0,
            /* *C: */       output,
            /* ldc: */      outHeight * outWidth
        );

        // Cycle cleanup
        delete[] srcIm2col;
        delete[] output;
    }

    // 结束计时
    auto end = std::clock();
    float time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;

    // Cleanup
    delete[] kernel2col;

    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < kernel_h; j++) {
            delete[] kernel[i][j];
        }
        delete[] kernel[i];
    }
    delete[] kernel;

    for (int i = 0; i < in_height; i++) {
        delete[] src[i];
    }
    delete[] src;

    return time_ms;
}


int main()
{
    float time_ms = measure_im2col_time(100);

    cout<<"im2colOrigin Total time cost: " << time_ms <<" ms"<<endl;

    return 0;
}
