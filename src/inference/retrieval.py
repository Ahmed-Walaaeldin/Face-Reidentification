from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def late_fusion_distance(dist_facenet: float, dist_vit: float, *, facenet_weight: float) -> float:
	"""Late fusion of distances (matches main.ipynb)."""

	return facenet_weight * dist_facenet + (1.0 - facenet_weight) * dist_vit


@torch.no_grad()
def build_reference_embeddings(
	*,
	facenet_model,
	vit_model,
	train_loader_facenet,
	train_loader_vit,
	label_encoder,
	device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
	"""Builds mean embedding per identity for both models (main.ipynb)."""

	facenet_model.eval()
	vit_model.eval()

	ref_embeddings_facenet: Dict[str, list] = {}
	ref_embeddings_vit: Dict[str, list] = {}

	for (images_facenet, labels), (images_vit, _) in zip(
		tqdm(train_loader_facenet, desc="FaceNet Ref"),
		train_loader_vit,
	):
		images_facenet = images_facenet.to(device)
		images_vit = images_vit.to(device)

		emb_f = facenet_model(images_facenet)
		emb_v = vit_model(images_vit)
		emb_f = nn.functional.normalize(emb_f, p=2, dim=1)
		emb_v = nn.functional.normalize(emb_v, p=2, dim=1)

		for ef, ev, lbl in zip(emb_f.cpu().numpy(), emb_v.cpu().numpy(), labels.numpy()):
			gt = label_encoder.inverse_transform([lbl])[0]
			if gt not in ref_embeddings_facenet:
				ref_embeddings_facenet[gt] = []
				ref_embeddings_vit[gt] = []
			ref_embeddings_facenet[gt].append(ef)
			ref_embeddings_vit[gt].append(ev)

	ref_mean_f = {gt: np.mean(v, axis=0) for gt, v in ref_embeddings_facenet.items()}
	ref_mean_v = {gt: np.mean(ref_embeddings_vit[gt], axis=0) for gt in ref_embeddings_vit}
	return ref_mean_f, ref_mean_v


@torch.no_grad()
def predict_with_late_fusion(
	*,
	facenet_model,
	vit_model,
	ref_embeddings_facenet: Dict[str, np.ndarray],
	ref_embeddings_vit: Dict[str, np.ndarray],
	test_loader_facenet,
	test_loader_vit,
	test_full_paths,
	device: torch.device,
	facenet_weight: float = 0.7,
	threshold: float = 0.28,
) -> list[dict]:
	"""Predicts identities for the test set with late-fusion distances (main.ipynb)."""

	facenet_model.eval()
	vit_model.eval()

	predictions = []
	total_predictions = 0
	doesnt_exist_count = 0

	for (images_facenet, _), (images_vit, _) in zip(
		tqdm(test_loader_facenet, desc="FaceNet Test"),
		test_loader_vit,
	):
		images_facenet = images_facenet.to(device)
		images_vit = images_vit.to(device)

		emb_f = facenet_model(images_facenet)
		emb_v = vit_model(images_vit)
		emb_f = nn.functional.normalize(emb_f, p=2, dim=1).cpu().numpy()
		emb_v = nn.functional.normalize(emb_v, p=2, dim=1).cpu().numpy()

		batch_paths = test_full_paths[total_predictions : total_predictions + len(emb_f)]
		for ef, ev, full_path in zip(emb_f, emb_v, batch_paths):
			distances_facenet = {gt: np.linalg.norm(ef - ref_emb) for gt, ref_emb in ref_embeddings_facenet.items()}
			distances_vit = {gt: np.linalg.norm(ev - ref_emb) for gt, ref_emb in ref_embeddings_vit.items()}

			combined_distances = {
				gt: late_fusion_distance(distances_facenet[gt], distances_vit[gt], facenet_weight=facenet_weight)
				for gt in distances_facenet
			}

			min_dist = min(combined_distances.values())
			closest_gt = min(combined_distances, key=combined_distances.get)

			if min_dist <= threshold:
				predicted_gt = closest_gt
			else:
				predicted_gt = "doesn't_exist"
				doesnt_exist_count += 1

			import os

			predictions.append({"image": os.path.basename(full_path), "gt": predicted_gt, "threshold": float(min_dist)})
			total_predictions += 1

	proportion = doesnt_exist_count / total_predictions if total_predictions > 0 else 0
	print("\nResults:")
	print(f"Total Predictions: {total_predictions}")
	print(f"Doesn't Exist Count: {doesnt_exist_count}")
	print(f"Doesn't Exist Proportion: {proportion:.4f}")
	return predictions

