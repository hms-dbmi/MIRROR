#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mirror.config import BIOMARKERS
from mirror.dataset import PatchDataset, collate_fn
from mirror.model import MIRROR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--biomarkers", nargs="+", default=None)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--name", default="mirror")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    biomarkers = list(args.biomarkers) if args.biomarkers else list(BIOMARKERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv, low_memory=False)
    if args.image_col not in df.columns:
        raise ValueError(f"no column {args.image_col!r}")

    os.makedirs(args.output, exist_ok=True)
    mirror = MIRROR.load(args.checkpoint, device=device)
    img_dataset = PatchDataset(df, image=args.image_col, transform=mirror.transform)
    img_loader = DataLoader(img_dataset, batch_size=args.batch_size, 
                num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)

    image_features_list = []
    for batch in tqdm(img_loader, desc="encode_image"):
        image_features_list.append(
            mirror.encode_image(batch["img"], use_proj=False, normalize=False).cpu()
        )
    image_features = torch.cat(image_features_list, dim=0)
    torch.save(image_features, os.path.join(args.output, f"{args.name}.pt"))

    image_embeddings = F.normalize(
        mirror.visual.proj_layer(image_features.to(device)),
        dim=-1,
    )

    pred_df = pd.DataFrame()
    pred_df["image_name"] = df[args.image_col].apply(
        lambda path: os.path.basename(str(path))
    )

    logit_scale = mirror.logit_scale.exp().item()
    for biomarker in biomarkers:
        biomarker_embeddings = mirror.encode_biomarker(biomarker).to(device)
        logits = image_embeddings @ biomarker_embeddings
        pred_df[biomarker] = (
            F.softmax(logits * logit_scale, dim=1).numpy()[:, 1]
        )

    pred_df["logit_scale"] = logit_scale
    pred_path = os.path.join(args.output, f"{args.name}_pred.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[MIRROR] wrote {pred_path} ({len(pred_df)} patches, {len(biomarkers)} biomarkers)")


if __name__ == "__main__":
    main()

