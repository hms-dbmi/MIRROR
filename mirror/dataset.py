"""Patch image dataset."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    imgs, ids = [], []
    for item in batch:
        img = item["img"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        imgs.append(img)
        ids.append(item["id"])
    return {"img": torch.stack(imgs), "id": torch.tensor(ids)}


class PatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image: str = "image_path",
        transform=None,
    ) -> None:
        self.paths = df[image].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = str(self.paths[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"img": img, "id": idx}
