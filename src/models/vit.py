from __future__ import annotations

import torch
import torch.nn as nn

from .losses import CurricularFace


def _require_timm():
	try:
		import timm  # noqa: F401
	except Exception as exc:  # pragma: no cover
		raise ImportError("timm is required for ViT models. Install with: pip install timm") from exc


class AdvancedFaceReIDModel(nn.Module):
	"""ViT + embedding head + (optional) CurricularFace + classifier.

	This matches the training model in ViT_finetuning.ipynb.
	"""

	def __init__(
		self,
		num_classes: int,
		embedding_dim: int = 1024,
		margin: float = 0.5,
		scale: float = 64.0,
		dropout: float = 0.6,
	):
		super().__init__()
		_require_timm()
		import timm

		self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
		self.embedding_layer = nn.Sequential(
			nn.Linear(768, 2048),
			nn.BatchNorm1d(2048),
			nn.PReLU(),
			nn.Dropout(dropout),
			nn.Linear(2048, embedding_dim),
		)

		self.curricularface = CurricularFace(
			in_features=embedding_dim,
			out_features=num_classes,
			scale=scale,
			margin=margin,
		)
		self.classifier = nn.Linear(embedding_dim, num_classes)

	def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
		features = self.backbone(x)
		embeddings = self.embedding_layer(features)
		classifier_logits = self.classifier(embeddings)
		curricular_logits = self.curricularface(embeddings, labels) if labels is not None else None
		return embeddings, curricular_logits, classifier_logits


class ViTEmbeddingModel(nn.Module):
	"""Embedding-only ViT model used in main.ipynb inference."""

	def __init__(self, embedding_dim: int = 1024, dropout: float = 0.5):
		super().__init__()
		_require_timm()
		import timm

		self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
		self.embedding_layer = nn.Sequential(
			nn.Linear(768, 2048),
			nn.BatchNorm1d(2048),
			nn.PReLU(),
			nn.Dropout(dropout),
			nn.Linear(2048, embedding_dim),
		)

	def forward(self, x: torch.Tensor):
		features = self.backbone(x)
		return self.embedding_layer(features)

