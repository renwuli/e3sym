#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyzd(b, m, 3) xyzd(b, n , 3)
__global__ void build_graph_kernel(int b, int n, int m, float thre, int nsample, const float *__restrict__ new_xyzd, const float *__restrict__ xyzd, int64_t *__restrict__ idx_out, float *__restrict__ dist_out, float * __restrict__ edge_out)
{
    int batch_index = blockIdx.x;
    xyzd += batch_index * m * 3;
    new_xyzd += batch_index * n * 3;
    idx_out += m * nsample * batch_index;
    dist_out += m * nsample * batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j = index; j < m; j += stride)
    {
        float new_x = new_xyzd[j * 3 + 0];
        float new_y = new_xyzd[j * 3 + 1];
        float new_z = new_xyzd[j * 3 + 2];
        float new_d = new_xyzd[j * 3 + 3];

        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            float x = xyzd[k * 3 + 0];
            float y = xyzd[k * 3 + 1];
            float z = xyzd[k * 3 + 2];
            float d = xyzd[k * 3 + 3];
            
            float plane_distance1 = sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y) + (z - new_z) * (z - new_z) + (d - new_d) * (d - new_d));

            float plane_distance2 = sqrt((x + new_x) * (x + new_x) + (y + new_y) * (y + new_y) + (z + new_z) * (z + new_z) + (d + new_d) * (d + new_d));

            float plane_distance = min(plane_distance1, plane_distance2);

            if (plane_distance < thre)
            {
                if (cnt == 0)
                {
                    for (int l = 0; l < nsample; ++l)
                    {
                        idx_out[j * nsample + l] = k;
                    }
                }
                idx_out[j * nsample + cnt] = k;
                dist_out[j * nsample + cnt] = plane_distance;
                edge_out[j * nsample + cnt] = 1.0 - (plane_distance);
                ++cnt;
            }
        }
    }
}


void build_graph_kernel_wrapper(int b, int n, int m, float thre, int nsample, const float *new_xyzd, const float *xyzd, int64_t *idx_out, float *dist_out, float *edge_out)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    build_graph_kernel<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, thre, nsample, new_xyzd, xyzd, idx_out, dist_out, edge_out);
    CUDA_CHECK_ERRORS();
}

__global__ void build_graph_batch_kernel(int size_x, int size_y, float thre, int nsample, const float* __restrict__ x, const float* __restrict__ y, const int64_t* __restrict__ batch_x, const int64_t* __restrict__ batch_y, int64_t* __restrict__ idx_out, float* __restrict__ dist_out, float* __restrict__ edge_out) {
    const ptrdiff_t batch_idx = blockIdx.x;
    const ptrdiff_t start_idx_x = batch_x[batch_idx];
    const ptrdiff_t start_idx_y = batch_y[batch_idx];
    const ptrdiff_t end_idx_x = batch_x[batch_idx + 1];
    const ptrdiff_t end_idx_y = batch_y[batch_idx + 1];

    for (ptrdiff_t n_y = start_idx_y + threadIdx.x; n_y < end_idx_y; n_y += blockDim.x)
    {
        int64_t cnt = 0;
        float yx = y[n_y * 4 + 0];
        float yy = y[n_y * 4 + 1];
        float yz = y[n_y * 4 + 2];
        float yd = y[n_y * 4 + 3];


        for (ptrdiff_t n_x = start_idx_x; n_x < end_idx_x; n_x++)
        {
            float xx = x[n_x * 4 + 0];
            float xy = x[n_x * 4 + 1];
            float xz = x[n_x * 4 + 2];
            float xd = x[n_x * 4 + 3];


            float plane_distance1 = sqrt((xx - yx) * (xx - yx) + (xy - yy) * (xy - yy) + (xz - yz) * (xz - yz) + (xd - yd) * (xd - yd));

            float plane_distance2 = sqrt((xx + yx) * (xx + yx) + (xy + yy) * (xy + yy) + (xz + yz) * (xz + yz)+ (xd + yd) * (xd + yd));

            float plane_distance = min(plane_distance1, plane_distance2);

            if (plane_distance < thre) 
            {
                idx_out[n_y * nsample + cnt] = n_x;
                dist_out[n_y * nsample + cnt] = plane_distance;
                edge_out[n_y * nsample + cnt] = 1.0 - plane_distance;
                cnt++;
            }
            if (cnt >= nsample)
            {
                break;
            }
        }
    }
}

void build_graph_batch_kernel_wrapper(int64_t batch_size, int size_x, int size_y, float thre, int nsample, const float* __restrict__ x, const float* __restrict__ y, const int64_t* __restrict__ batch_x, const int64_t* __restrict__ batch_y, int64_t* __restrict__ idx_out, float* __restrict__ dist_out, float* __restrict__ edge_out)
{
    build_graph_batch_kernel<<<batch_size, TOTAL_THREADS_SPARSE>>>(size_x, size_y, thre, nsample, x, y, batch_x, batch_y, idx_out, dist_out, edge_out);

    CUDA_CHECK_ERRORS();
}
