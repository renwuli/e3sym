import numpy as np
import pyvista as pv
import matplotlib.cm as cm
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from prefetch_generator import BackgroundGenerator


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=-1)


def reproducible(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("Conv2d") == -1:
        m.weight.data.normal_(mean, std)
    elif classname.find("BatchNorm") != -1 and classname.find("BatchNorm2d") == -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)


def weight_init_linear_zeros(m):
    if type(m) == nn.Linear:
        m.weight.data.zero_()
        m.bias.data.zero_()


def get_choice(batch, n, m):
    choice = np.arange(0, n)
    np.random.shuffle(choice)
    choice = torch.from_numpy(choice[:m]).long()
    return choice.unsqueeze_(0).unsqueeze_(0).expand(batch, 3, -1)


def random_subsample(points, m):
    # points: [b, 3, n]
    # m: number of points to be sampled, m < n
    batch, _, n = points.size()
    choice = get_choice(batch, n, m).to(points.device)
    return torch.gather(points, dim=2, index=choice)


def random_sample(points, n):
    # points: [m, 3]
    choice = torch.randperm(points.shape[0]).to(points.device)
    new_points = points[choice[:n], :]

    if points.shape[0] < n:
        k = n - points.size(0)
        choice = torch.randint(points.size(0), (k,))
        new_points = torch.cat([new_points, points[choice]])
    return new_points


def prepare_points(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points.shape[0] == 3:
        points = points.transpose()
    return points


def o3d_pointcloud(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points.shape[0] == 3:
        points = points.transpose()

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


def read_point_cloud(path):
    pointcloud = o3d.io.read_point_cloud(path)
    points = np.array(pointcloud.points).astype(np.float32)
    return points


def save_pointcloud(path, points):
    pointcloud = o3d_pointcloud(points)
    o3d.io.write_point_cloud(path, pointcloud)


def read_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh


def obj2ply(obj_path, ply_path):
    mesh = read_mesh(obj_path)
    o3d.io.write_triangle_mesh(ply_path, mesh)


def save_plane2points(plane, path):
    """
    plane: [m, 4]
    """
    normal = plane[:, :3]
    d = plane[:, 3]

    points = normal * d[:, None]

    points = prepare_points(points)

    save_pointcloud(path, points)


def set_print():
    torch.set_printoptions(
        precision=None,
        threshold=None,
        edgeitems=None,
        linewidth=None,
        profile=None,
        sci_mode=False,
    )


def count_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k
