#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor> build_graph(at::Tensor new_xyz, at::Tensor xyz, const float thre, const int nsample);
std::tuple<at::Tensor, at::Tensor, at::Tensor> build_graph_batch(at::Tensor x, at::Tensor y, at::Tensor batch_x, at::Tensor batch_y, const float thre, const int nsample);
