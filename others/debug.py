from fall_detection.models import CustomYOLO

import torch

def _make_grid(nx=20, ny=20, i=0):
    """Generate detection grid and anchor grid."""
    shape = 1, 8, ny, nx, 2  # grid shape
    t = torch.float32
    na = 8
    y, x = torch.arange(ny), torch.arange(nx)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
    # anchor_grid = (self.anchors[i, :na] * 64).view((1, na, 1, 1, 2)).expand(shape)
    return grid

def _make_grid1(nx=20, ny=20, i=0):
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    grid = torch.stack((xv, yv), 2).float().view(1, 1, ny, nx, 2)
    return grid

def test_make_grid():
    grid1 = _make_grid()
    grid2 = _make_grid1()
    assert torch.allclose(grid1, grid2 - 0.5), "Grid generation mismatch"

# test_make_grid()
model = CustomYOLO("configs/model/custom_yolo.yaml", "detect", verbose=True)
print(model)
model = model.eval()
data = torch.rand(1, 3, 640, 640)
output = model.model(data)
breakpoint()
print(output)


