from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import FaceDataset
from src.data.loaders import build_weighted_sampler
from src.data.transforms import build_facenet_eval_transform, build_facenet_train_transform_stage2
from src.models.facenet import FaceNetModel
from src.models.losses import TripletLoss, generate_triplets_stage2
from src.training.checkpointing import load_checkpoint_state, save_checkpoint_state
from src.training.evaluation import evaluate_embeddings_clustering_stage2
from src.training.loops import seed_everything
from src.training.config_utils import deep_merge, interpolate_config, is_windows, load_yaml, repo_root, resolve_path


class Config:
	# Copied from FaceNet_finetuning.ipynb (stage 2 overrides)
	BASE_DIR = "/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification"
	TRAIN_CSV = os.path.join(BASE_DIR, "trainset.csv")
	CHECKPOINT_DIR = "/kaggle/working/checkpoints/facenet_triplet"
	BATCH_SIZE = 512
	NUM_EPOCHS = 34
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	EMBEDDING_DIM = 1024
	IMAGE_SIZE = (160, 160)
	NUM_WORKERS = 2
	SCALING_FACTOR = 1.0
	NUM_CLASSES = None
	START_EPOCH = 24
	CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_epoch_24.pth")
	_OPTIMIZER = {}
	_SCHEDULER = {}


def _apply_yaml_config(model_cfg_path: Path, data_cfg_path: Path) -> None:
	base = repo_root()
	model_cfg = load_yaml(model_cfg_path)
	data_cfg = load_yaml(data_cfg_path)
	cfg = interpolate_config(deep_merge(data_cfg, model_cfg))

	data = cfg.get("data", {})
	facenet = cfg.get("facenet", {})
	optimizer = cfg.get("optimizer", {})
	scheduler = cfg.get("scheduler", {})

	Config.BASE_DIR = str(resolve_path(data.get("base_dir", ""), base=base))
	Config.TRAIN_CSV = str(resolve_path(data.get("train_csv", ""), base=base))
	Config.CHECKPOINT_DIR = str(resolve_path(facenet.get("checkpoint_dir", Config.CHECKPOINT_DIR), base=base))
	Config.BATCH_SIZE = int(facenet.get("batch_size", Config.BATCH_SIZE))
	Config.NUM_EPOCHS = int(facenet.get("num_epochs", Config.NUM_EPOCHS))
	Config.START_EPOCH = int(facenet.get("start_epoch", Config.START_EPOCH))
	Config.EMBEDDING_DIM = int(facenet.get("embedding_dim", Config.EMBEDDING_DIM))
	image_size = facenet.get("image_size", list(Config.IMAGE_SIZE))
	Config.IMAGE_SIZE = (int(image_size[0]), int(image_size[1]))
	Config.NUM_WORKERS = int(facenet.get("num_workers", 0 if is_windows() else Config.NUM_WORKERS))
	Config.SCALING_FACTOR = float(facenet.get("scaling_factor", Config.SCALING_FACTOR))

	checkpoint_path = facenet.get("checkpoint_path", "")
	if checkpoint_path:
		Config.CHECKPOINT_PATH = str(resolve_path(checkpoint_path, base=base))

	Config._OPTIMIZER = optimizer
	Config._SCHEDULER = scheduler

	if is_windows() and Config.NUM_WORKERS != 0:
		print("[facenet_stage2] Windows detected: forcing num_workers=0 for stability")
		Config.NUM_WORKERS = 0


def _build_optimizer(model):
	opt = getattr(Config, "_OPTIMIZER", {}) or {}
	name = str(opt.get("name", "AdamW"))
	lr = float(opt.get("lr", 5e-5))
	weight_decay = float(opt.get("weight_decay", 1e-2))
	if name.lower() != "adamw":
		raise ValueError(f"Unsupported optimizer {name}. Only AdamW is implemented.")
	return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer):
	sch = getattr(Config, "_SCHEDULER", {}) or {}
	name = str(sch.get("name", "CosineAnnealingLR"))
	if name.lower() == "cosineannealinglr":
		t_max = int(sch.get("t_max", 10))
		eta_min = float(sch.get("eta_min", 1e-6))
		return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
	if name.lower() == "multisteplr":
		milestones = sch.get("milestones", [10, 20, 30])
		gamma = float(sch.get("gamma", 0.5))
		return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
	raise ValueError(f"Unsupported scheduler {name}")


def _unique_path_if_exists(path: str) -> str:
	p = Path(path)
	if not p.exists():
		return str(p)
	stem = p.stem
	suffix = p.suffix
	parent = p.parent
	for i in range(1, 10_000):
		cand = parent / f"{stem}_run{i}{suffix}"
		if not cand.exists():
			return str(cand)
	return str(parent / f"{stem}_run{os.getpid()}{suffix}")


def get_margin(epoch: int) -> float:
	return 2.0 if epoch >= Config.START_EPOCH else 1.6


def train_step(model, images, labels, *, epoch: int, batch_idx: int, optimizer, criterion_triplet: TripletLoss, train_loader):
	optimizer.zero_grad()
	embeddings = model(images)
	hard_ratio = min(0.05 * epoch, 0.9)
	anchor, positive, negative = generate_triplets_stage2(
		embeddings,
		labels,
		margin=criterion_triplet.margin,
		device=Config.DEVICE,
		hard_ratio=hard_ratio,
	)
	if anchor is None:
		return 0.0, 0.0, 0.0

	triplet_loss, mean_pos_dist, mean_neg_dist = criterion_triplet(anchor, positive, negative)
	l2_reg = 0.001 * torch.norm(embeddings, p=2)
	pos_target = max(0.8 - 0.03 * epoch, 0.4)
	pos_penalty = 0.5 * (F.relu(mean_pos_dist - pos_target) + F.relu(0.05 - mean_pos_dist))
	loss = triplet_loss + l2_reg + pos_penalty

	if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
		print(
			f"Epoch {epoch+1}, Batch {batch_idx} - Triplet Loss: {triplet_loss.item():.4f}, "
			f"Mean Pos Dist: {float(mean_pos_dist):.4f}, Mean Neg Dist: {float(mean_neg_dist):.4f}, "
			f"Embeddings Mean: {float(embeddings.mean()):.4f}, Std: {float(embeddings.std()):.4f}"
		)

	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
	optimizer.step()
	return float(loss.item()), float(mean_pos_dist.item()), float(mean_neg_dist.item())


def train_model(model, train_loader, val_loader, criterion_triplet: TripletLoss, optimizer, scheduler, *, start_epoch: int, num_epochs: int):
	best_acc = 98.90
	best_epoch = start_epoch
	prev_neg_dist = 0.0
	plateau_count = 0

	if os.path.exists(Config.CHECKPOINT_PATH):
		checkpoint = load_checkpoint_state(Config.CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=True)
		if "model_state_dict" in checkpoint:
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			model.load_state_dict(checkpoint)
		print(f"Loaded checkpoint from {Config.CHECKPOINT_PATH} (epoch {Config.START_EPOCH})")
	else:
		raise FileNotFoundError(f"Checkpoint {Config.CHECKPOINT_PATH} not found!")

	for epoch in range(start_epoch, num_epochs):
		model.train()
		train_loss = 0.0
		total_pos_dist = 0.0
		total_neg_dist = 0.0

		criterion_triplet.margin = get_margin(epoch)
		print(f"Epoch {epoch+1} - Margin: {criterion_triplet.margin:.2f}")

		for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
			images = images.to(Config.DEVICE)
			labels = labels.to(Config.DEVICE)
			loss, pos_dist, neg_dist = train_step(
				model,
				images,
				labels,
				epoch=epoch,
				batch_idx=batch_idx,
				optimizer=optimizer,
				criterion_triplet=criterion_triplet,
				train_loader=train_loader,
			)
			train_loss += loss
			total_pos_dist += pos_dist
			total_neg_dist += neg_dist

		train_loss /= max(len(train_loader), 1)
		avg_pos_dist = total_pos_dist / max(len(train_loader), 1)
		avg_neg_dist = total_neg_dist / max(len(train_loader), 1)
		print(
			f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, "
			f"Avg Pos Dist: {avg_pos_dist:.4f}, Avg Neg Dist: {avg_neg_dist:.4f}"
		)

		mean_pos_dist, mean_neg_dist, max_pos_dist, val_acc, _ = evaluate_embeddings_clustering_stage2(
			model,
			val_loader,
			num_classes=Config.NUM_CLASSES,
			device=Config.DEVICE,
			criterion_triplet=criterion_triplet,
		)

		if val_acc > best_acc:
			best_acc = val_acc
			best_epoch = epoch + 1
			out_path = _unique_path_if_exists(f"{Config.CHECKPOINT_DIR}/best_model_epoch_{best_epoch}.pth")
			save_checkpoint_state(out_path, model_state_dict=model.state_dict())
			print(f"Saved best model at epoch {best_epoch} with Clustering Accuracy: {val_acc:.2f}%")

		scheduler.step()
		print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

		if not np.isnan(mean_neg_dist) and abs(mean_neg_dist - prev_neg_dist) < 0.01:
			plateau_count += 1
			if plateau_count >= 7:
				print(f"Plateau detected for {plateau_count} epochs")
		else:
			plateau_count = 0
		prev_neg_dist = mean_neg_dist if not np.isnan(mean_neg_dist) else prev_neg_dist

		if epoch > start_epoch + 6 and val_acc < 99.0 and plateau_count >= 7:
			print(f"Early stopping at epoch {epoch+1}: Accuracy plateaued at {val_acc:.2f}%")
			break

	return best_acc, best_epoch


def main():
	parser = argparse.ArgumentParser(description="Train FaceNet stage-2 (continue from checkpoint) via YAML configs")
	parser.add_argument(
		"--model-config",
		default=str(repo_root() / "configs" / "facenet" / "stage2.yaml"),
		help="Path to FaceNet stage-2 YAML config (default: configs/facenet/stage2.yaml)",
	)
	parser.add_argument(
		"--data-config",
		default=str(repo_root() / "configs" / "data" / "retail_faces.yaml"),
		help="Path to dataset YAML config (default: configs/data/retail_faces.yaml)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Load configs + CSV and validate resume checkpoint exists, then exit without training",
	)
	args = parser.parse_args()

	seed_everything(42)
	_apply_yaml_config(Path(args.model_config), Path(args.data_config))
	os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

	print("Loading training data...")
	train_df = pd.read_csv(
		Config.TRAIN_CSV,
		sep=",",
		engine="python",
		header=0,
		names=["image_path", "gt"],
		on_bad_lines="skip",
	)
	train_df = train_df.dropna()
	train_df["image_path"] = train_df["image_path"].str.strip()
	train_df["gt"] = train_df["gt"].str.strip()
	train_df["full_path"] = train_df["image_path"].apply(lambda x: os.path.join(Config.BASE_DIR, x))
	if args.dry_run:
		if not os.path.exists(Config.CHECKPOINT_PATH):
			raise FileNotFoundError(f"[facenet_stage2] Resume checkpoint not found: {Config.CHECKPOINT_PATH}")
		missing = (~train_df["full_path"].apply(os.path.exists)).sum()
		print(
			f"[facenet_stage2] Dry-run OK. Resume: {Config.CHECKPOINT_PATH}. Samples: {len(train_df)}, missing_paths: {int(missing)}"
		)
		return None, 0.0, None, train_df
	le = LabelEncoder()
	train_df["label"] = le.fit_transform(train_df["gt"])
	Config.NUM_CLASSES = len(le.classes_)
	print(f"Training data loaded: {len(train_df)} samples, {Config.NUM_CLASSES} classes")

	train_data, val_data = train_test_split(
		train_df,
		test_size=0.2,
		stratify=train_df["label"],
		random_state=42,
	)

	train_transform = build_facenet_train_transform_stage2(Config.IMAGE_SIZE)
	val_transform = build_facenet_eval_transform(Config.IMAGE_SIZE)

	train_dataset = FaceDataset(train_data, transform=train_transform, is_train=True)
	val_dataset = FaceDataset(val_data, transform=val_transform, is_train=True)

	sampler = build_weighted_sampler(train_data["label"].values, num_classes=Config.NUM_CLASSES)

	train_loader = DataLoader(
		train_dataset,
		batch_size=Config.BATCH_SIZE,
		sampler=sampler,
		num_workers=Config.NUM_WORKERS,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=Config.BATCH_SIZE,
		shuffle=False,
		num_workers=Config.NUM_WORKERS,
		pin_memory=True,
	)

	model = FaceNetModel(embedding_dim=Config.EMBEDDING_DIM, scaling_factor=Config.SCALING_FACTOR).to(Config.DEVICE)
	criterion_triplet = TripletLoss(margin=2.0)
	optimizer = _build_optimizer(model)
	scheduler = _build_scheduler(optimizer)

	best_acc, best_epoch = train_model(
		model,
		train_loader,
		val_loader,
		criterion_triplet,
		optimizer,
		scheduler,
		start_epoch=Config.START_EPOCH,
		num_epochs=Config.NUM_EPOCHS,
	)
	print(f"Best clustering accuracy: {best_acc:.2f}% at epoch {best_epoch}")
	return model, best_acc, le, train_df


if __name__ == "__main__":
	main()

