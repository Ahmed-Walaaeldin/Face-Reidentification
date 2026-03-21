from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm


def _require_sklearn_and_scipy():
	try:
		from sklearn.cluster import KMeans  # noqa: F401
		from scipy.stats import mode  # noqa: F401
	except Exception as exc:  # pragma: no cover
		raise ImportError(
			"scikit-learn and scipy are required for clustering evaluation. "
			"Install with: pip install scikit-learn scipy"
		) from exc


def evaluate_embeddings_clustering(model, val_loader, *, num_classes: int, device: torch.device):
	"""Stage-1 FaceNet clustering accuracy evaluation (as in notebook)."""

	_require_sklearn_and_scipy()
	from sklearn.cluster import KMeans
	from scipy.stats import mode

	model.eval()
	all_embeddings = []
	all_labels = []
	with torch.no_grad():
		for images, labels in tqdm(val_loader, desc="Generating validation embeddings"):
			images = images.to(device)
			embeddings = model(images)
			all_embeddings.append(embeddings.cpu().numpy())
			all_labels.append(np.asarray(labels))

	all_embeddings = np.concatenate(all_embeddings)
	all_labels = np.concatenate(all_labels)

	embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
	dist_matrix = torch.cdist(torch.tensor(embeddings_norm), torch.tensor(embeddings_norm)).numpy()
	pos_mask = (all_labels[:, None] == all_labels[None, :]) & ~np.eye(len(all_labels), dtype=bool)
	neg_mask = all_labels[:, None] != all_labels[None, :]
	pos_dists = dist_matrix[pos_mask]
	neg_dists = dist_matrix[neg_mask]
	mean_pos_dist = float(np.mean(pos_dists))
	mean_neg_dist = float(np.mean(neg_dists))
	max_pos_dist = float(np.max(pos_dists))

	kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(embeddings_norm)
	pred_labels = kmeans.labels_
	label_map = {}
	for cluster in range(num_classes):
		cluster_labels = all_labels[pred_labels == cluster]
		if len(cluster_labels) > 0:
			mode_result = mode(cluster_labels)
			label_map[cluster] = mode_result.mode.item()
	mapped_labels = [label_map.get(pred, -1) for pred in pred_labels]
	accuracy = float(np.mean([1 if pred == true else 0 for pred, true in zip(mapped_labels, all_labels)]) * 100)

	print(
		f"Validation - Mean Pos Dist: {mean_pos_dist:.4f}, Mean Neg Dist: {mean_neg_dist:.4f}, "
		f"Max Pos Dist: {max_pos_dist:.4f}, Clustering Accuracy: {accuracy:.2f}%"
	)

	return mean_pos_dist, mean_neg_dist, max_pos_dist, accuracy


def evaluate_embeddings_clustering_stage2(model, val_loader, *, num_classes: int, device: torch.device, criterion_triplet):
	"""Stage-2 FaceNet clustering + triplet loss evaluation (as in notebook)."""

	_require_sklearn_and_scipy()
	from sklearn.cluster import KMeans
	from scipy.stats import mode
	from src.models.losses import TripletLoss, generate_triplets_stage2

	model.eval()
	all_embeddings = []
	all_labels = []
	val_triplet_loss_total = 0.0
	num_val_batches = 0
	val_criterion = TripletLoss(margin=0.8)

	with torch.no_grad():
		for images, labels in tqdm(val_loader, desc="Generating validation embeddings"):
			images = images.to(device)
			labels = labels.to(device)
			embeddings = model(images)
			anchor, positive, negative = generate_triplets_stage2(
				embeddings,
				labels,
				margin=criterion_triplet.margin,
				device=device,
				hard_ratio=0.0,
			)
			if anchor is not None:
				val_triplet_loss = val_criterion(anchor, positive, negative)[0]
				val_triplet_loss_total += float(val_triplet_loss.item())
				num_val_batches += 1
			all_embeddings.append(embeddings.cpu().numpy())
			all_labels.append(labels.cpu().numpy())

	all_embeddings = np.concatenate(all_embeddings)
	all_labels = np.concatenate(all_labels)
	avg_val_triplet_loss = val_triplet_loss_total / num_val_batches if num_val_batches > 0 else 0.0

	embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
	dist_matrix = torch.cdist(torch.tensor(embeddings_norm), torch.tensor(embeddings_norm)).numpy()
	pos_mask = (all_labels[:, None] == all_labels[None, :]) & ~np.eye(len(all_labels), dtype=bool)
	neg_mask = all_labels[:, None] != all_labels[None, :]
	pos_dists = dist_matrix[pos_mask]
	neg_dists = dist_matrix[neg_mask]
	mean_pos_dist = float(np.mean(pos_dists)) if len(pos_dists) > 0 else 0.0
	mean_neg_dist = float(np.mean(neg_dists)) if len(neg_dists) > 0 else float("nan")
	max_pos_dist = float(np.max(pos_dists)) if len(pos_dists) > 0 else 0.0

	kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10).fit(embeddings_norm)
	pred_labels = kmeans.labels_
	label_map = {}
	for cluster in range(num_classes):
		cluster_labels = all_labels[pred_labels == cluster]
		if len(cluster_labels) > 0:
			mode_result = mode(cluster_labels)
			label_map[cluster] = mode_result.mode.item()
	mapped_labels = [label_map.get(pred, -1) for pred in pred_labels]
	accuracy = float(np.mean([1 if pred == true else 0 for pred, true in zip(mapped_labels, all_labels)]) * 100)

	print(
		f"Validation - Mean Pos Dist: {mean_pos_dist:.4f}, Mean Neg Dist: {mean_neg_dist:.4f}, "
		f"Max Pos Dist: {max_pos_dist:.4f}, Clustering Accuracy: {accuracy:.2f}%, Avg Val Triplet Loss: {avg_val_triplet_loss:.4f}"
	)
	return mean_pos_dist, mean_neg_dist, max_pos_dist, accuracy, avg_val_triplet_loss

