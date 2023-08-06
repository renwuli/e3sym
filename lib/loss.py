import torch
import torch.nn as nn

from pytorch3d.loss import chamfer_distance


class SymmetryDistanceError(nn.Module):
    def __init__(self):
        super(SymmetryDistanceError, self).__init__()

    def _distance(self, pos, plane):
        """
        pos: [n, 3]
        plane: [4]
        """

        normal = plane[:3]
        d = plane[3]

        ref_pos = pos - 2 * (torch.sum(pos * normal, dim=1) + d).unsqueeze(-1) * normal
        return chamfer_distance(ref_pos.unsqueeze(0), pos.unsqueeze(0))[0]

    def forward(self, pos, plane, batch):
        """
        pos:    [b, n, 3]
        plane:  [m, 4]
        batch:  [m]
        """
        b, n, _ = pos.size()
        m = plane.size(0)
        assert batch.size(0) == m

        loss = torch.tensor(0.0, device=pos.device)
        for i in range(m):
            cur_plane = plane[i]
            cur_batch = batch[i]
            cur_pos = pos[cur_batch]
            loss.add_(self._distance(cur_pos, cur_plane))

        loss.div_(m)
        return loss
