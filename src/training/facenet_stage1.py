from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import FaceDataset
from src.data.loaders import build_weighted_sampler
from src.data.transforms import build_facenet_eval_transform, build_facenet_train_transform
from src.models.facenet import FaceNetModel
from src.models.losses import TripletLoss, generate_triplets_stage1
from src.training.checkpointing import save_checkpoint_state
from src.training.evaluation import evaluate_embeddings_clustering
from src.training.loops import seed_everything
from src.training.config_utils import deep_merge, interpolate_config, is_windows, load_yaml, repo_root, resolve_path


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


class Config:
	# Copied from FaceNet_finetuning.ipynb (stage 1)
	BASE_DIR = "/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification"
	TRAIN_CSV = os.path.join(BASE_DIR, "trainset.csv")
	CHECKPOINT_DIR = "/kaggle/working/checkpoints/facenet_triplet"
	BATCH_SIZE = 512
	NUM_EPOCHS = 45
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	EMBEDDING_DIM = 1024
	IMAGE_SIZE = (160, 160)
	NUM_WORKERS = 2
	SCALING_FACTOR = 500.0
	NUM_CLASSES = None


def _apply_yaml_config(model_cfg_path: Path, data_cfg_path: Path) -> None:
	base = repo_root()
	model_cfg = load_yaml(model_cfg_path)
	data_cfg = load_yaml(data_cfg_path)
	cfg = interpolate_config(deep_merge(data_cfg, model_cfg))

	data = cfg.get("data", {})
	paths = cfg.get("paths", {})
	facenet = cfg.get("facenet", {})
	optimizer = cfg.get("optimizer", {})
	scheduler = cfg.get("scheduler", {})

	Config.BASE_DIR = str(resolve_path(data.get("base_dir", ""), base=base))
	Config.TRAIN_CSV = str(resolve_path(data.get("train_csv", ""), base=base))
	Config.CHECKPOINT_DIR = str(resolve_path(facenet.get("checkpoint_dir", ""), base=base))

	Config.BATCH_SIZE = int(facenet.get("batch_size", Config.BATCH_SIZE))
	Config.NUM_EPOCHS = int(facenet.get("num_epochs", Config.NUM_EPOCHS))
	Config.EMBEDDING_DIM = int(facenet.get("embedding_dim", Config.EMBEDDING_DIM))
	image_size = facenet.get("image_size", list(Config.IMAGE_SIZE))
	Config.IMAGE_SIZE = (int(image_size[0]), int(image_size[1]))
	Config.NUM_WORKERS = int(facenet.get("num_workers", 0 if is_windows() else Config.NUM_WORKERS))
	Config.SCALING_FACTOR = float(facenet.get("scaling_factor", Config.SCALING_FACTOR))

	# Optional override if someone wants to point directly at an image root.
	_ = paths.get("image_dir", None)

	Config._OPTIMIZER = optimizer
	Config._SCHEDULER = scheduler

	if is_windows() and Config.NUM_WORKERS != 0:
		print("[facenet_stage1] Windows detected: forcing num_workers=0 for stability")
		Config.NUM_WORKERS = 0


def _build_optimizer(model):
	opt = getattr(Config, "_OPTIMIZER", {}) or {}
	name = str(opt.get("name", "AdamW"))
	lr = float(opt.get("lr", 3e-4))
	weight_decay = float(opt.get("weight_decay", 1e-4))
	if name.lower() != "adamw":
		raise ValueError(f"Unsupported optimizer {name}. Only AdamW is implemented.")
	return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer):
	sch = getattr(Config, "_SCHEDULER", {}) or {}
	name = str(sch.get("name", "MultiStepLR"))
	if name.lower() == "multisteplr":
		milestones = sch.get("milestones", [10, 20, 30])
		gamma = float(sch.get("gamma", 0.5))
		return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
	if name.lower() == "cosineannealinglr":
		t_max = int(sch.get("t_max", 10))
		eta_min = float(sch.get("eta_min", 1e-6))
		return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
	raise ValueError(f"Unsupported scheduler {name}")


def get_margin(epoch: int) -> float:
	if epoch < 4:
		return 1.0
	if epoch < 7:
		return 2.0
	return 3.0


def train_step(model, images, labels, *, epoch: int, batch_idx: int, optimizer, criterion_triplet: TripletLoss, train_loader):
	optimizer.zero_grad()
	embeddings = model(images)
	anchor, positive, negative = generate_triplets_stage1(
		embeddings,
		labels,
		margin=criterion_triplet.margin,
		device=Config.DEVICE,
		hard_ratio=0.75,
	)
	if anchor is None:
		return 0.0, 0.0, 0.0
	loss, mean_pos_dist, mean_neg_dist = criterion_triplet(anchor, positive, negative)
	if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
		print(
			f"Epoch {epoch+1}, Batch {batch_idx} - Triplet Loss: {loss.item():.4f}, "
			f"Mean Pos Dist: {mean_pos_dist.item():.4f}, Mean Neg Dist: {mean_neg_dist.item():.4f}, "
			f"Embeddings Mean: {embeddings.mean().item():.4f}, Std: {embeddings.std().item():.4f}"
		)
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
	optimizer.step()
	return float(loss.item()), float(mean_pos_dist.item()), float(mean_neg_dist.item())


def train_model(model, train_loader, val_loader, criterion_triplet: TripletLoss, optimizer, scheduler, num_epochs: int):
	best_acc = 0.0
	best_epoch = 0

	for epoch in range(num_epochs):
		model.train()
		train_loss = 0.0
		total_pos_dist = 0.0
		total_neg_dist = 0.0

		criterion_triplet.margin = get_margin(epoch + 1)
		print(f"Epoch {epoch+1} - Margin: {criterion_triplet.margin}")

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

		_, _, _, val_acc = evaluate_embeddings_clustering(model, val_loader, num_classes=Config.NUM_CLASSES, device=Config.DEVICE)

		if val_acc > best_acc:
			best_acc = val_acc
			best_epoch = epoch + 1
			out_path = _unique_path_if_exists(f"{Config.CHECKPOINT_DIR}/best_model_epoch_{best_epoch}.pth")
			save_checkpoint_state(out_path, model_state_dict=model.state_dict())
			print(f"Saved best model at epoch {best_epoch} with Clustering Accuracy: {val_acc:.2f}%")

		scheduler.step()

	return best_acc, best_epoch


def main():
	parser = argparse.ArgumentParser(description="Train FaceNet stage-1 (triplet) from YAML configs")
	parser.add_argument(
		"--model-config",
		default=str(repo_root() / "configs" / "facenet" / "stage1.yaml"),
		help="Path to FaceNet stage-1 YAML config (default: configs/facenet/stage1.yaml)",
	)
	parser.add_argument(
		"--data-config",
		default=str(repo_root() / "configs" / "data" / "retail_faces.yaml"),
		help="Path to dataset YAML config (default: configs/data/retail_faces.yaml)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Load configs + CSV and validate paths, then exit without training",
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
		missing = (~train_df["full_path"].apply(os.path.exists)).sum()
		print(f"[facenet_stage1] Dry-run OK. Samples: {len(train_df)}, missing_paths: {int(missing)}")
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
	print(f"Train split: {len(train_data)} samples, Val split: {len(val_data)} samples")

	train_transform = build_facenet_train_transform(Config.IMAGE_SIZE)
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
	criterion_triplet = TripletLoss(margin=1.0)
	optimizer = _build_optimizer(model)
	scheduler = _build_scheduler(optimizer)

	best_acc, best_epoch = train_model(model, train_loader, val_loader, criterion_triplet, optimizer, scheduler, Config.NUM_EPOCHS)
	print(f"Best clustering accuracy: {best_acc:.2f}% at epoch {best_epoch}")
	return model, best_acc, le, train_df


if __name__ == "__main__":
	main()

