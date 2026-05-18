"""MIRROR: Zero-shot prediction of spatial immune profiles from routine pathology slides using a vision–language model."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import load_checkpoint
from timm.layers import SwiGLUPacked
import timm

from mirror.config import TEMPLATES


class Virchow2Encoder(nn.Module):
    """H&E image encoder (Virchow2)."""

    def __init__(
        self,
        model_name: str = "hf-hub:paige-ai/Virchow2",
        device: str = "cpu",
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        image_model = timm.create_model(
            model_name,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        self.output_dim = output_dim
        self.base_model = image_model.to(device)
        self.proj_layer = nn.Linear(image_model.embed_dim, output_dim).to(device)
        self.use_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_model(x)
        embedding = output[:, 0]
        if self.use_proj:
            embedding = self.proj_layer(embedding)
        return embedding


class MIRROR:
    """MIRROR: Zero-shot prediction of spatial immune profiles from routine pathology slides using a vision–language model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        transform: Any,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = device

    @property
    def visual(self) -> Virchow2Encoder:
        return self.model.visual

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "MIRROR":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model, _, clip_transform = open_clip.create_model_and_transforms("ViT-L-14")
        clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        clip_model.visual = Virchow2Encoder(
            output_dim=clip_model.visual.output_dim,
            device=str(device),
        )

        incompatible = load_checkpoint(
            clip_model, path, strict=False, weights_only=True, device="cpu"
        )
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing:
            print(f"[MIRROR] missing keys ({len(missing)}), e.g. {missing[:3]}")
        if unexpected:
            print(f"[MIRROR] unexpected keys ({len(unexpected)}), e.g. {unexpected[:3]}")

        clip_model.eval()
        clip_model.to(device)
        return cls(
            clip_model,
            transform=clip_transform,
            tokenizer=clip_tokenizer,
            device=device,
        )

    @torch.no_grad()
    def encode_image(
        self,
        images: torch.Tensor,
        *,
        use_proj: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        images = images.to(self.device)
        if use_proj:
            self.visual.use_proj = True
            feats = self.model.encode_image(images, normalize=False)
        else:
            self.visual.use_proj = False
            feats = self.model.encode_image(images, normalize=False)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def encode_text(
        self,
        text: Union[str, Sequence[str], torch.Tensor],
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        if isinstance(text, torch.Tensor):
            token_ids = text.to(self.device)
        else:
            if isinstance(text, str):
                text = [text]
            token_ids = self.tokenizer(list(text)).to(self.device)
        return self.model.encode_text(token_ids, normalize=normalize)

    @torch.no_grad()
    def encode_biomarker(
        self,
        biomarker: str,
        templates: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Text embeddings for ``{biomarker} Negative`` and ``{biomarker} Positive`` ([D, 2])."""
        if templates is None:
            templates = TEMPLATES
        classnames = [[f"{biomarker} Negative"], [f"{biomarker} Positive"]]
        weights = []
        for classnames_for_class in classnames:
            embeddings_for_class = []
            for classname in classnames_for_class:
                texts = [t.replace("CLASSNAME", classname) for t in templates]
                emb = self.encode_text(texts, normalize=False)
                embeddings_for_class.append(F.normalize(emb, dim=-1))
            class_embedding = torch.stack(embeddings_for_class, dim=0).mean(dim=(0, 1))
            class_embedding = class_embedding / class_embedding.norm()
            weights.append(class_embedding)
        return torch.stack(weights, dim=1)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale
