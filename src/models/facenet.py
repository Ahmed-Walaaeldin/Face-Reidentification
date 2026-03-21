from __future__ import annotations

import torch
import torch.nn as nn


class FaceNetModel(nn.Module):
	"""FaceNet embedding model from FaceNet_finetuning.ipynb and main.ipynb."""

	def __init__(self, embedding_dim: int = 1024, scaling_factor: float = 500.0):
		super().__init__()
		try:
			from facenet_pytorch import InceptionResnetV1
		except Exception as exc:  # pragma: no cover
			raise ImportError(
				"facenet-pytorch is required for FaceNetModel. "
				"Install it with: pip install facenet-pytorch==2.5.3"
			) from exc

		self.backbone = InceptionResnetV1(pretrained="vggface2")
		self.backbone.classify = False
		self.embedding_layer = nn.Sequential(
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(1024, embedding_dim),
			nn.BatchNorm1d(embedding_dim),
		)
		self.scaling_factor = float(scaling_factor)

	def forward(self, x):
		features = self.backbone(x)
		embeddings = self.embedding_layer(features)
		return embeddings * self.scaling_factor

