#include "op.h"
#include "compat.h"
#include "utils.h"

void build_graph_kernel_wrapper(int b, int n, int m, float thre, int nsample, const float* new_xyz, const float* xyz, int64_t* idx, float* dist_out, float* edge_out);
void build_graph_batch_kernel_wrapper(int64_t batch_size, int size_x, int size_y, float thre, int nsample, const float* x, const float* y, const int64_t* batch_x, const int64_t* batch_y, int64_t* idx_out, float* dist_out, float* edge_out);

std::tuple<at::Tensor, at::Tensor, at::Tensor> build_graph(at::Tensor new_xyz, at::Tensor xyz, const float thre, const int nsample)
{
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    CHECK_CUDA(xyz);
    CHECK_CUDA(new_xyz);

    at::Tensor idx = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                                  at::device(new_xyz.device()).dtype(at::ScalarType::Long));
    at::Tensor dist = torch::full({new_xyz.size(0), new_xyz.size(1), nsample}, -1,
                                  at::device(new_xyz.device()).dtype(at::ScalarType::Float));
    at::Tensor edge = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                                  at::device(new_xyz.device()).dtype(at::ScalarType::Float));

    build_graph_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1), thre, nsample,
                                          new_xyz.DATA_PTR<float>(), xyz.DATA_PTR<float>(),
                                          idx.DATA_PTR<int64_t>(), dist.DATA_PTR<float>(), edge.DATA_PTR<float>());

    return std::make_tuple(idx, dist, edge);
}

at::Tensor degree(at::Tensor row, int64_t num_nodes)
{
    auto zero = at::zeros(num_nodes, row.options());
    auto one = at::ones(row.size(0), row.options());
    return zero.scatter_add_(0, row, one);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> build_graph_batch(at::Tensor x, at::Tensor y, at::Tensor batch_x, at::Tensor batch_y, const float thre, const int nsample) {
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(y);
    CHECK_IS_FLOAT(x);
    CHECK_IS_FLOAT(y);
    CHECK_CUDA(x);
    CHECK_CUDA(y);
    CHECK_CUDA(batch_x);
    CHECK_CUDA(batch_y);

    at::Tensor idx = torch::full({y.size(0), nsample}, -1, at::device(y.device()).dtype(at::ScalarType::Long));
    at::Tensor dist = torch::full({y.size(0), nsample}, -1, at::device(y.device()).dtype(at::ScalarType::Float));
    at::Tensor edge = torch::zeros({y.size(0), nsample}, at::device(y.device()).dtype(at::ScalarType::Float));

    cudaSetDevice(x.get_device());
    auto batch_sizes = (int64_t*) malloc(sizeof(int64_t));
    cudaMemcpy(batch_sizes, batch_x[-1].DATA_PTR<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);
    auto batch_size = batch_sizes[0] + 1;
    
    batch_x = degree(batch_x, batch_size);
    batch_x = at::cat({at::zeros(1, batch_x.options()), batch_x.cumsum(0)}, 0);
    batch_y = degree(batch_y, batch_size);
    batch_y = at::cat({at::zeros(1, batch_y.options()), batch_y.cumsum(0)}, 0);

    build_graph_batch_kernel_wrapper(batch_size, x.size(0), y.size(0), thre, nsample, x.DATA_PTR<float>(), y.DATA_PTR<float>(), batch_x.DATA_PTR<int64_t>(), batch_y.DATA_PTR<int64_t>(), idx.DATA_PTR<int64_t>(), dist.DATA_PTR<float>(), edge.DATA_PTR<float>());

    return std::make_tuple(idx, dist, edge);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_graph", &build_graph, "build graph");
    m.def("build_graph_batch", &build_graph_batch, "build graph batch");
}
