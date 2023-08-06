import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import ball_query


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def generate_24_rotations():
    res = []
    for id in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        R = np.identity(3)[:, id].astype(np.float32)
        R1 = np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
        R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
        R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    for id in [[0, 2, 1], [1, 0, 2], [2, 1, 0]]:
        R = np.identity(3)[:, id].astype(np.float32)
        R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
        R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
        R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
        R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
        res += [R1, R2, R3, R4]
    res = np.stack(res, axis=0)
    res = torch.from_numpy(res).float()
    return res


def generate_8_rotations():
    res = []
    R = np.identity(3).astype(np.float32)
    R1 = np.asarray([R[:, 0], R[:, 1], R[:, 2]]).T
    R2 = np.asarray([-R[:, 0], -R[:, 1], R[:, 2]]).T
    R3 = np.asarray([-R[:, 0], R[:, 1], -R[:, 2]]).T
    R4 = np.asarray([R[:, 0], -R[:, 1], -R[:, 2]]).T
    res += [R1, R2, R3, R4]
    R = np.identity(3).astype(np.float32)
    R1 = np.asarray([-R[:, 0], -R[:, 1], -R[:, 2]]).T
    R2 = np.asarray([-R[:, 0], R[:, 1], R[:, 2]]).T
    R3 = np.asarray([R[:, 0], -R[:, 1], R[:, 2]]).T
    R4 = np.asarray([R[:, 0], R[:, 1], -R[:, 2]]).T
    res += [R1, R2, R3, R4]
    res = np.stack(res, axis=0)
    res = torch.from_numpy(res).float()
    return res


def pca(points):
    # points: [b, m, k, 3]
    cov = torch.matmul(points.transpose(2, 3), points)
    cov = cov / points.shape[2]
    # cov: [b, m, 3, 3]
    e, v = torch.linalg.eigh(cov)
    # e: [b, m, 3], v: [b, m, 3, 3]
    can_points = torch.matmul(points, v)
    # can_points: [b, m, k, 3]
    return can_points


def group(points, dense_points, K=64, radius=0.15):
    # points: [b, n, 3]
    nn_dists, nn_indices, nn_points = ball_query(
        points, dense_points, K=K, radius=radius, return_nn=True
    )
    # nn_points: [b, n, k, 3]
    nn_points = nn_points - points[:, :, None, :]  # [b, n, k, 3]
    nn_points = pca(nn_points)
    return nn_points


class SoftTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1, sigma: float = 0.0001):
        super(SoftTopK, self).__init__()
        self.k = k
        self.num_samples = num_samples
        self.sigma = sigma

    def __call__(self, x):
        return SoftTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class SoftTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1, sigma: float = 0.0001):
        """
        the sigma lower, the topk harder
        """
        b, n, m = x.size()
        noise = torch.normal(mean=0.0, std=1.0, size=(b, n, num_samples, m)).to(
            dtype=x.dtype, device=x.device
        )
        perturbed_x = x.unsqueeze(2) + noise * sigma  # [b, n, nsample, m]
        indices = torch.topk(
            perturbed_x, k=k, dim=-1, sorted=False, largest=False
        ).indices
        perturbed_output = F.one_hot(
            indices, num_classes=m
        ).float()  # [b, n, nsample, k, m]
        indicators = perturbed_output.mean(dim=2)  # [b, n, k, m]

        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnxkm,bnxm->bnkm", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bnkm,bnkm->bnm", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)
