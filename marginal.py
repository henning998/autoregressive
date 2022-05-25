#!pip install -q pytorch_lightning

from re import X
import time

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
#from tqdm.auto import tqdm
from tqdm import tqdm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import pytorch_lightning as pl


print(matplotlib.get_backend())
matplotlib.use("GTK3Agg")


def build_conv_layer(fi, fo, k=3, upscale=False, pool=False, last=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if upscale else nn.Identity(),
        nn.Conv2d(fi, fo, kernel_size=3, padding=k//2),
        nn.ReLU(inplace=True) if not last else nn.Identity(),
        nn.BatchNorm2d(fo) if not last else nn.Identity(),
        nn.MaxPool2d(2, 2) if pool else nn.Identity(),
    )


class BaseModel(pl.LightningModule):
    """small encoder-decoder cnn"""
    n_pts: int
    n_channels: int

    def __init__(self, left=(8, 16), right=(16, 8), lr=1e-3):
        super().__init__()
        self.lr = lr
        left = self.n_channels, *left
        right = *right, self.n_pts
        assert len(left) == len(right)
        layers = [build_conv_layer(fi, fo, pool=True)
                  for fi, fo in zip(left, left[1:])]
        layers += [build_conv_layer(fi, fo, upscale=True)
                   for fi, fo in zip(right, right[1:-1])]
        layers += [build_conv_layer(right[-2],
                                    right[-1], upscale=True, last=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):  # (B, n_channels, h, w)
        return self.layers(x)  # (B, n_pts, h, w)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class FunDataset(torch.utils.data.Dataset):
    """dataset from a specified function"""

    def __init__(self, data_fun, epoch_size=5000):
        self.data_fun = data_fun
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        return self.data_fun()


def get_dataloader(data_fun, bs=16):
    return torch.utils.data.DataLoader(
        FunDataset(data_fun),
        batch_size=bs,
        worker_init_fn=lambda *_: np.random.seed(None),
        num_workers=2,
    )


def get_trainer(max_epochs=10):
    return pl.Trainer(
        gpus=[0],
        logger=False, enable_checkpointing=False,
        max_epochs=max_epochs,
    )


def circle_data_fun(res=128, radius=10, n_pts=2):
    theta_pts_obj = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    center = np.random.randint(radius, res - radius, 2)
    theta = np.random.rand() * np.pi * 2

    img = np.zeros((res, res), dtype=np.float32)
    cv2.circle(img, tuple(center), radius, color=1., thickness=-1)

    theta_cam_pts = theta + theta_pts_obj
    pts = center + np.array((
        np.cos(theta_cam_pts),
        np.sin(theta_cam_pts)
    )).T * radius
    return img, np.round(pts).astype(int)


img, pts = circle_data_fun()
plt.imshow(img, cmap='gray')
plt.scatter(*pts.T, c=[(1, 0, 0), (0, 0, 1)])
plt.show()


class MarginalModel(BaseModel):
    n_pts = 2
    n_channels = 1

    def training_step(self, batch, _):
        img, pts = batch  # (B, h, w), (B, 2, 2xy), (B,)
        B, h, w = img.shape
        xx, yy = pts.permute(2, 0, 1)
        target = yy * w + xx  # (B, 2)

        lgts = self.forward(img[:, None])  # (B, n_pts, h, w)
        lgts = lgts.view(B, 2, -1)

        loss = F.cross_entropy(
            lgts.view(B * self.n_pts, -1),
            target.view(B * self.n_pts)
        )
        return loss


model = MarginalModel()
get_trainer(max_epochs=10).fit(model, get_dataloader(circle_data_fun))


n_rows, n_cols, size = 3, 5, 3
axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*size, n_rows*size))[1]
model.eval()
with torch.no_grad():
    for ax_col in axs.T:
        img, pts = circle_data_fun()
        h, w = img.shape
        lgts = model.forward(torch.from_numpy(img)[None, None])[
            0]  # (n_pts, h, w)
        probs = F.softmax(lgts.view(2, -1), dim=1).view(2, h, w)
        ax_col[0].imshow(img, cmap='gray')
        ax_col[0].scatter(*pts.T, c=[(1, 0, 0), (0, 0, 1)])
        ax_col[1].imshow(probs[0], cmap='Reds')
        ax_col[2].imshow(probs[1], cmap='Blues')
for ax in axs.reshape(-1):
    ax.axis('off')
plt.tight_layout()

#######################################################################################


def joint_data_fun(model):
    model.eval()
    with torch.no_grad():
        img, pts = circle_data_fun()
        h, w = img.shape
        lgts = model.forward(torch.from_numpy(img)[None, None])[
            0]  # (n_pts, h, w)
        probs = F.softmax(lgts.view(2, -1), dim=1).view(2, h, w)
        maxproppos = probs[0].argmax
        print(maxproppos)
    return img, pts


class JointModel(BaseModel):
    n_pts = 1
    n_channels = 2

    def joint_traning_step(self, batch):
        img, pts = batch
        print(img.shape)
        B, h, w, c = img.shape
        xx, yy = pts.permute(2, 0, 1)
        target = yy[1] * w + xx[1]

        lgts = self.forward(img[:,None])
        print(lgts.shape)
        lgts = lgts.view(B,1,-1)

        loss = F.cross_entropy(
            lgts.view(B,-1),  # times 2 maybe
            target.view(B)
        )
        return loss

model_joint = JointModel()
get_trainer(max_epochs=10).fit(model_joint,get_dataloader(joint_data_fun))