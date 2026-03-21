from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import AdvancedRetailFaceDataset
from src.data.loaders import build_weighted_sampler
from src.data.transforms import build_vit_eval_transform, build_vit_train_transform
from src.models.vit import AdvancedFaceReIDModel
from src.training.config_utils import deep_merge, interpolate_config, is_windows, load_yaml, repo_root, resolve_path


class Config:
	# Copied from ViT_finetuning.ipynb (stage 2)
	BASE_DIR = "/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification"
	TRAIN_CSV = os.path.join(BASE_DIR, "trainset.csv")
	TEST_CSV = os.path.join(BASE_DIR, "eval_set.csv")
	IMAGE_DIR = BASE_DIR
	CHECKPOINT_DIR = "/kaggle/input/vitevenstricterbase/finetune_checkpoints"
	FINETUNE_CHECKPOINT_DIR = "/kaggle/working/finetune_checkpoints"

	BATCH_SIZE = 32
	NUM_EPOCHS = 20
	LEARNING_RATE = 3e-5
	EMBEDDING_DIM = 1024
	NUM_FOLDS = 5
	MARGIN = 0.85
	SCALE = 64.0
	TRIPLET_MARGIN = 0.85
	IMG_SIZE = 224
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	NUM_WORKERS = 2


def _apply_yaml_config(model_cfg_path: Path, data_cfg_path: Path) -> None:
	base = repo_root()
	model_cfg = load_yaml(model_cfg_path)
	data_cfg = load_yaml(data_cfg_path)
	cfg = interpolate_config(deep_merge(data_cfg, model_cfg))

	data = cfg.get("data", {})
	vit = cfg.get("vit", {})
	curricularface = cfg.get("curricularface", {})
	metric_learning = cfg.get("metric_learning", {})

	Config.BASE_DIR = str(resolve_path(data.get("base_dir", ""), base=base))
	Config.TRAIN_CSV = str(resolve_path(data.get("train_csv", ""), base=base))
	Config.TEST_CSV = str(resolve_path(data.get("test_csv", ""), base=base))
	Config.IMAGE_DIR = str(resolve_path(vit.get("image_dir", data.get("base_dir", "")), base=base))

	Config.CHECKPOINT_DIR = str(resolve_path(vit.get("checkpoint_dir", Config.CHECKPOINT_DIR), base=base))
	Config.FINETUNE_CHECKPOINT_DIR = str(resolve_path(vit.get("finetune_checkpoint_dir", Config.FINETUNE_CHECKPOINT_DIR), base=base))

	Config.BATCH_SIZE = int(vit.get("batch_size", Config.BATCH_SIZE))
	Config.NUM_EPOCHS = int(vit.get("num_epochs", Config.NUM_EPOCHS))
	Config.LEARNING_RATE = float(vit.get("learning_rate", Config.LEARNING_RATE))
	Config.EMBEDDING_DIM = int(vit.get("embedding_dim", Config.EMBEDDING_DIM))
	Config.NUM_FOLDS = int(vit.get("num_folds", Config.NUM_FOLDS))
	Config.IMG_SIZE = int(vit.get("image_size", Config.IMG_SIZE))
	Config.NUM_WORKERS = int(vit.get("num_workers", 0 if is_windows() else Config.NUM_WORKERS))

	Config.MARGIN = float(curricularface.get("margin", Config.MARGIN))
	Config.SCALE = float(curricularface.get("scale", Config.SCALE))
	Config.TRIPLET_MARGIN = float(metric_learning.get("triplet_margin", Config.TRIPLET_MARGIN))

	if is_windows() and Config.NUM_WORKERS != 0:
		print("[vit_stage2] Windows detected: forcing num_workers=0 for stability")
		Config.NUM_WORKERS = 0


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


def _require_metric_learning():
	try:
		from pytorch_metric_learning import miners, losses  # noqa: F401
		from torchmetrics import Accuracy  # noqa: F401
	except Exception as exc:  # pragma: no cover
		raise ImportError(
			"ViT training requires pytorch-metric-learning and torchmetrics. "
			"Install with: pip install pytorch_metric_learning torchmetrics"
		) from exc


def finetune_fold(fold, train_idx, val_idx, df, num_classes: int):
	_require_metric_learning()
	from pytorch_metric_learning import miners, losses
	from torchmetrics import Accuracy

	model = AdvancedFaceReIDModel(
		num_classes=num_classes,
		embedding_dim=Config.EMBEDDING_DIM,
		margin=Config.MARGIN,
		scale=Config.SCALE,
		dropout=0.6,
	).to(Config.DEVICE)

	optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
	scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")

	checkpoint_path = f"{Config.CHECKPOINT_DIR}/fold_{fold+1}_finetuned.pth"
	finetune_checkpoint_path = _unique_path_if_exists(
		f"{Config.FINETUNE_CHECKPOINT_DIR}/fold_{fold+1}_aggressive_finetuned.pth"
	)

	start_epoch = 0
	if os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
		model.load_state_dict(checkpoint["model_state"])
		optimizer.load_state_dict(checkpoint["optimizer_state"])
		scaler_state = checkpoint.get("scaler_state", {})
		if scaler_state:
			scaler.load_state_dict(scaler_state)
		else:
			print(f"No valid scaler state in checkpoint for fold {fold+1}, using fresh scaler")
		start_epoch = checkpoint["epoch"] + 1
		print(f"Loaded fine-tuned checkpoint for fold {fold+1} from epoch {checkpoint['epoch']}")
	else:
		raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

	# Notebook resets start_epoch to 0 after loading
	start_epoch = 0

	train_labels = df.iloc[train_idx]["label"].values
	class_counts = np.bincount(train_labels, minlength=num_classes)
	class_weights = 1.0 / (class_counts + 1e-6)
	class_weights = class_weights / class_weights.sum()

	ce_criterion = nn.CrossEntropyLoss(
		weight=torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE),
		label_smoothing=0.1,
	)
	triplet_criterion = losses.TripletMarginLoss(margin=Config.TRIPLET_MARGIN)
	contrastive_criterion = losses.ContrastiveLoss(pos_margin=0.1, neg_margin=1.0)

	sampler = build_weighted_sampler(train_labels, num_classes=num_classes)

	train_dataset = AdvancedRetailFaceDataset(
		df.iloc[train_idx],
		transform=build_vit_train_transform(finetune=True, image_size=Config.IMG_SIZE),
	)
	train_loader = DataLoader(
		train_dataset,
		batch_size=Config.BATCH_SIZE,
		sampler=sampler,
		num_workers=Config.NUM_WORKERS,
	)

	val_dataset = AdvancedRetailFaceDataset(
		df.iloc[val_idx],
		transform=build_vit_eval_transform(image_size=Config.IMG_SIZE),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=Config.BATCH_SIZE,
		shuffle=False,
		num_workers=Config.NUM_WORKERS,
	)

	train_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(Config.DEVICE)
	val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(Config.DEVICE)

	miner = miners.MultiSimilarityMiner()
	best_acc = 0.0

	os.makedirs(Config.FINETUNE_CHECKPOINT_DIR, exist_ok=True)

	with tqdm(total=Config.NUM_EPOCHS, desc=f"Finetuning Fold {fold+1}", leave=True) as epoch_pbar:
		for epoch in range(start_epoch, start_epoch + Config.NUM_EPOCHS):
			model.train()
			total_loss = 0.0
			with tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + Config.NUM_EPOCHS}", leave=False) as train_pbar:
				for images, labels in train_pbar:
					if images is None or labels is None:
						continue
					images = images.to(Config.DEVICE)
					labels = labels.to(Config.DEVICE)
					optimizer.zero_grad(set_to_none=True)
					with torch.cuda.amp.autocast(enabled=Config.DEVICE.type == "cuda"):
						embeddings, curricular_logits, classifier_logits = model(images, labels)
						ce_loss = ce_criterion(classifier_logits, labels)
						curricular_loss = ce_criterion(curricular_logits, labels)
						hard_pairs = miner(embeddings, labels)
						triplet_loss = triplet_criterion(embeddings, labels, hard_pairs)
						contrastive_loss = contrastive_criterion(embeddings, labels)
						loss = ce_loss + curricular_loss + triplet_loss + contrastive_loss
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
					total_loss += float(loss.item())
					preds = torch.argmax(classifier_logits, dim=1)
					train_acc_metric.update(preds, labels)
					train_pbar.set_postfix({"loss": f"{total_loss / (train_pbar.n + 1):.4f}"})

			train_acc = float(train_acc_metric.compute().item())
			train_acc_metric.reset()
			scheduler.step()

			model.eval()
			with torch.no_grad():
				with torch.cuda.amp.autocast(enabled=Config.DEVICE.type == "cuda"):
					for images, labels in val_loader:
						if images is None or labels is None:
							continue
						images = images.to(Config.DEVICE)
						labels = labels.to(Config.DEVICE)
						_, _, classifier_logits = model(images)
						preds = torch.argmax(classifier_logits, dim=1)
						val_acc_metric.update(preds, labels)

			val_acc = float(val_acc_metric.compute().item())
			val_acc_metric.reset()

			epoch_pbar.set_postfix(
				{
					"Train Acc": f"{train_acc:.4f}",
					"Loss": f"{total_loss / max(len(train_loader), 1):.4f}",
					"Val Acc": f"{val_acc:.4f}",
				}
			)
			epoch_pbar.update(1)

			if val_acc > best_acc:
				best_acc = val_acc
				torch.save(
					{
						"model_state": model.state_dict(),
						"optimizer_state": optimizer.state_dict(),
						"scaler_state": scaler.state_dict(),
						"epoch": epoch,
					},
					finetune_checkpoint_path,
				)
				print(
					f"Saved aggressive fine-tuned checkpoint for fold {fold+1} at epoch {epoch} with accuracy {best_acc * 100:.2f}%"
				)

	return best_acc


def main_stage_2():
	parser = argparse.ArgumentParser(description="Train ViT stage-2 (aggressive finetune) via YAML configs")
	parser.add_argument(
		"--model-config",
		default=str(repo_root() / "configs" / "vit" / "stage2.yaml"),
		help="Path to ViT stage-2 YAML config (default: configs/vit/stage2.yaml)",
	)
	parser.add_argument(
		"--data-config",
		default=str(repo_root() / "configs" / "data" / "retail_faces.yaml"),
		help="Path to dataset YAML config (default: configs/data/retail_faces.yaml)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Load configs + CSV and validate input fold checkpoints exist, then exit without training",
	)
	args = parser.parse_args()
	_apply_yaml_config(Path(args.model_config), Path(args.data_config))

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
	le = LabelEncoder()
	train_df["label"] = le.fit_transform(train_df["gt"])
	valid_paths = train_df["full_path"].apply(os.path.exists)
	if not valid_paths.all():
		invalid_paths = train_df[~valid_paths]["full_path"].tolist()
		print(f"Warning: {len(invalid_paths)} invalid training paths found: {invalid_paths[:5]}")
		train_df = train_df[valid_paths]
	print(f"Total valid training samples: {len(train_df)}")
	if args.dry_run:
		missing = (~train_df["full_path"].apply(os.path.exists)).sum()
		missing_ckpts = []
		for fold in range(1, Config.NUM_FOLDS + 1):
			p = Path(Config.CHECKPOINT_DIR) / f"fold_{fold}_finetuned.pth"
			if not p.exists():
				missing_ckpts.append(str(p))
		print(f"[vit_stage2] Dry-run OK. Samples: {len(train_df)}, missing_paths: {int(missing)}")
		if missing_ckpts:
			print(f"[vit_stage2] Missing fold checkpoints: {missing_ckpts[:3]}" + (" ..." if len(missing_ckpts) > 3 else ""))
		return []

	os.makedirs(Config.FINETUNE_CHECKPOINT_DIR, exist_ok=True)
	skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
	fold_accuracies = []

	for fold, (train_idx, val_idx) in tqdm(
		enumerate(skf.split(train_df, train_df["gt"])),
		total=Config.NUM_FOLDS,
		desc="Finetuning All Folds",
	):
		print(f"\nFinetuning Fold {fold+1}")
		acc = finetune_fold(fold, train_idx, val_idx, train_df, len(train_df["gt"].unique()))
		fold_accuracies.append(acc)

	print(f"Average Fine-tuned Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
	return fold_accuracies


if __name__ == "__main__":
	main_stage_2()

