#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", default=None)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--biomarkers", nargs="+", required=True)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--image_col", default="image_name")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--name", default="mirror")
    args = parser.parse_args()

    pred_path = args.pred_csv or os.path.join(args.output, f"{args.name}_pred.csv")

    predictions = pd.read_csv(pred_path, low_memory=False)
    labels = pd.read_csv(args.label_csv, low_memory=False)

    if "image_name" not in predictions.columns:
        raise ValueError("pred_csv needs image_name")
    if args.image_col not in labels.columns:
        if "image_path" in labels.columns:
            labels = labels.copy()
            labels["image_name"] = labels["image_path"].apply(os.path.basename)
        else:
            raise ValueError(f"labels need {args.image_col!r} or image_path")

    predictions = predictions.set_index("image_name")
    labels = labels.set_index("image_name")
    common = predictions.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("no matching image_name")

    for biomarker in args.biomarkers:
        if biomarker not in predictions.columns:
            raise ValueError(f"no column {biomarker!r} in pred_csv")
        label_column = args.label_col or biomarker
        y_true = pd.to_numeric(labels.loc[common, label_column], errors="coerce").to_numpy()
        y_score = pd.to_numeric(predictions.loc[common, biomarker], errors="coerce").to_numpy()
        valid = np.isfinite(y_true) & np.isfinite(y_score)
        try:
            auc = float(roc_auc_score(y_true[valid], y_score[valid]))
        except ValueError:
            auc = float("nan")
        print(f"{biomarker}: AUROC={auc:.4f}  n={int(valid.sum())}")


if __name__ == "__main__":
    main()
