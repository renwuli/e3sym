import os
import random
import numpy as np
from tqdm import tqdm
import pyvista as pv

import torch

from lib.dataset import ShapeNetEval
from lib.model import SymmetryNet
from lib.loss import SymmetryDistanceError
from lib.util import DataLoaderX
from lib.ops.function import *


def save_planes(points, planes, save_path):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    raidus = sorted(np.amax(points, 0).tolist())

    plotter = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    plotter.add_points(points)

    if planes is not None:
        if isinstance(planes, torch.Tensor):
            planes = planes.detach().cpu().numpy()

        for i in range(len(planes)):
            plane = pv.Plane(
                center=(-planes[i, :3] * planes[i, 3]).tolist(),
                direction=[planes[i, 0], planes[i, 1], planes[i, 2]],
                i_size=0.9,
                j_size=0.9,
                i_resolution=2,
                j_resolution=2,
            )
            plotter.add_mesh(plane, show_edges=False, color="green", opacity=0.3)

    plotter.export_gltf(save_path)
    plotter.close()


@torch.no_grad()
def test(opts, path):
    random.seed(opts.seed)
    os.environ["PYTHONHASHSEED"] = str(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    model = SymmetryNet(
        opts.mlps,
        opts.ks,
        opts.radius,
        opts.rotations,
        opts.thre,
        opts.nsample,
        opts.min_cluster_size,
    ).cuda()

    if opts.weights:
        print(">>> loading pretrained weights")
        model.load_state_dict(torch.load(opts.weights))

    model.eval()

    dataset = ShapeNetEval(opts.eval_root, opts.npoints)
    data_loader = DataLoaderX(
        dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False
    )

    batch_size = opts.batch_size
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        pos = data
        pos = pos.cuda()

        cluster_plane, cluster_batch = model(pos)
        cluster_plane = cluster_plane.detach().cpu().numpy()
        cluster_batch = cluster_batch.detach().cpu().numpy()

        for j in range(batch_size):
            planes = cluster_plane[cluster_batch == j]
            points = pos[j]
            save_planes(points, planes, save_path=f"{path}/{i * batch_size + j}.gltf")


if __name__ == "__main__":
    import yaml
    import argparse
    from easydict import EasyDict
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    args = parser.parse_args()

    f = open(f"{args.config}")
    opt = EasyDict(yaml.safe_load(f))

    output_dir = "result"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    test(opt, output_dir)
