# MIRROR: Zero-shot prediction of spatial immune profiles from routine pathology slides using a vision–language model

## Abstract

Spatial proteomic imaging enables high-resolution characterization of the tumor immune microenvironment, facilitating biomarker discovery for therapeutic response and prognosis. However, its clinical translation remains bottlenecked by high costs, complex workflows, and limited scalability. In contrast, routine hematoxylin and eosin (H&E) slides capture rich morphological features, allowing advanced Artificial Intelligence (AI) models to infer underlying immune states. Here, we present MIRROR (Multiplexed Immune Representations via Routine Omni-tissue/On-Slide Readout), a zero-shot vision-language model that predicts multiplexed spatial immune profiles directly from H&E slides. We pretrained MIRROR on 732 thousand curated image–text pairs derived from co-registered H&E and spatial proteomic images of the same tissue sections. We evaluated MIRROR across 11 independent cohorts spanning 22 cancer types, comprising over 1 million patches profiled using 6 spatial proteomic imaging and immunohistochemical techniques: Orion, CODEX, TSA-based mIF, CyCIF, chromogenic mIHC, and IHC. Our model demonstrates robust zero-shot biomarker classification with strong cross-platform generalizability, achieving AUROCs up to 0.90 in the discovery cohorts and 0.95 in the independent cohorts. Predicted scores show high correlation with positive nuclei counts and strong spatial concordance with biomarker expression in mIF images. In addition, MIRROR-derived scores enhance survival prediction in stage I–II colorectal, lung, and breast cancers in independent cohorts (C-index gain up to 8.7 percentage points; log-rank test P-value = 0.006). Overall, MIRROR provides a scalable and cost-effective approach for high-plex immune profiling using standard H&E slides, bridging advanced spatial proteomics with routine clinical practice.

Place the model weights at `ckpt/mirror.pt` (or pass any path with `--checkpoint`).

## Usage

```python
from PIL import Image
import torch
import torch.nn.functional as F

from mirror import MIRROR

mirror = MIRROR.load("ckpt/mirror.pt")
image = mirror.transform(Image.open("patch.png").convert("RGB")).unsqueeze(0)
```

### Prompts

```python
prompts = [
    "an H&E patch of CD45 positive",
    "an H&E patch of CD45 negative",
]

with torch.inference_mode():
    image_emb = mirror.encode_image(image)
    text_emb = mirror.encode_text(prompts)  # [2, D]
    scale = mirror.logit_scale.exp()
    probs = F.softmax(image_emb @ text_emb.T * scale, dim=-1)[0]

for prompt, p in zip(prompts, probs.cpu().tolist()):
    print(f"{prompt}: {p:.3f}")
```

### Biomarkers

Same scoring, but only pass marker names—MIRROR builds the positive / negative prompts for you:

```python
biomarkers = ["CD45", "CD8a", "CD3e"]

with torch.inference_mode():
    image_emb = mirror.encode_image(image)
    scale = mirror.logit_scale.exp()
    probs = {}
    for bio in biomarkers:
        text_emb = mirror.encode_biomarker(bio)  # [D, 2]
        logits = image_emb @ text_emb
        probs[bio] = F.softmax(logits * scale, dim=-1)[0, 1].item()

print(probs)  # {'CD45': 0.71, 'CD8a': 0.23, 'CD3e': 0.88}
```