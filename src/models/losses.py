from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
	"""Triplet loss implementation used in FaceNet_finetuning.ipynb."""

	def __init__(self, margin: float):
		super().__init__()
		self.margin = float(margin)

	def forward(self, anchor, positive, negative):
		anchor = F.normalize(anchor, p=2, dim=1)
		positive = F.normalize(positive, p=2, dim=1)
		negative = F.normalize(negative, p=2, dim=1)
		pos_dist = F.pairwise_distance(anchor, positive)
		neg_dist = F.pairwise_distance(anchor, negative)
		loss = F.relu(pos_dist - neg_dist + self.margin)
		return loss.mean(), pos_dist.mean(), neg_dist.mean()


def generate_triplets_stage1(embeddings, labels, *, margin: float, device: torch.device, hard_ratio: float = 0.75):
	"""Triplet mining logic from FaceNet_finetuning.ipynb (stage 1)."""

	labels = labels.detach()
	anchor, positive, negative = [], [], []
	batch_size = embeddings.size(0)
	dist_matrix = torch.cdist(embeddings, embeddings)

	for i in range(batch_size):
		label = labels[i]
		pos_mask = (labels == label) & (torch.arange(batch_size, device=device) != i)
		neg_mask = labels != label

		pos_indices = torch.where(pos_mask)[0]
		neg_indices = torch.where(neg_mask)[0]
		if len(pos_indices) == 0 or len(neg_indices) == 0:
			continue

		pos_dists = dist_matrix[i, pos_indices]
		neg_dists = dist_matrix[i, neg_indices]

		pos_idx = pos_indices[torch.argmax(pos_dists)]
		pos_dist = pos_dists.max()

		if torch.rand(1).item() < hard_ratio:
			neg_idx = neg_indices[torch.argmin(neg_dists)]
		else:
			semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + margin)
			semi_hard_indices = neg_indices[semi_hard_mask]
			if len(semi_hard_indices) > 0:
				neg_idx = semi_hard_indices[torch.randint(0, len(semi_hard_indices), (1,)).item()]
			else:
				neg_idx = neg_indices[torch.argmin(neg_dists)]

		anchor.append(embeddings[i])
		positive.append(embeddings[pos_idx])
		negative.append(embeddings[neg_idx])

	if len(anchor) == 0:
		return None, None, None

	return torch.stack(anchor), torch.stack(positive), torch.stack(negative)


def generate_triplets_stage2(embeddings, labels, *, margin: float, device: torch.device, hard_ratio: float):
	"""Triplet mining logic from FaceNet_finetuning.ipynb (stage 2)."""

	labels = labels.detach()
	anchor, positive, negative = [], [], []
	batch_size = embeddings.size(0)
	dist_matrix = torch.cdist(embeddings, embeddings)

	for i in range(batch_size):
		label = labels[i]
		pos_mask = (labels == label) & (torch.arange(batch_size, device=device) != i)
		neg_mask = labels != label

		pos_indices = torch.where(pos_mask)[0]
		neg_indices = torch.where(neg_mask)[0]
		if len(pos_indices) == 0 or len(neg_indices) == 0:
			continue

		pos_dists = dist_matrix[i, pos_indices]
		neg_dists = dist_matrix[i, neg_indices]

		pos_idx = pos_indices[torch.argmax(pos_dists)]
		pos_dist = pos_dists.max()

		semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + margin)
		semi_hard_indices = neg_indices[semi_hard_mask]
		if len(semi_hard_indices) > 0 and torch.rand(1).item() >= hard_ratio:
			neg_idx = semi_hard_indices[torch.randint(0, len(semi_hard_indices), (1,)).item()]
		else:
			neg_idx = neg_indices[torch.argmin(neg_dists)]

		anchor.append(embeddings[i])
		positive.append(embeddings[pos_idx])
		negative.append(embeddings[neg_idx])

	if len(anchor) == 0:
		return None, None, None

	return torch.stack(anchor), torch.stack(positive), torch.stack(negative)


class CurricularFace(nn.Module):
	"""CurricularFace head from ViT_finetuning.ipynb."""

	def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5):
		super().__init__()
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)
		self.scale = float(scale)
		self.margin = float(margin)

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
		embeddings = F.normalize(embeddings, p=2, dim=1)
		weight = F.normalize(self.weight, p=2, dim=1)
		cos_theta = torch.matmul(embeddings, weight.T)

		margin_tensor = torch.tensor(self.margin, dtype=cos_theta.dtype, device=cos_theta.device)
		cos_m = torch.cos(margin_tensor)
		sin_m = torch.sin(margin_tensor)
		threshold = torch.cos(torch.tensor(np.pi, dtype=cos_theta.dtype, device=cos_theta.device) - margin_tensor)
		mm = sin_m * margin_tensor

		sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-6)
		cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
		mask = cos_theta > threshold
		cos_theta_m = torch.where(mask, cos_theta - mm, cos_theta_m)

		one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float().to(cos_theta.device)
		return self.scale * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)

