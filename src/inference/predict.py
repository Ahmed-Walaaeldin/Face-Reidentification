from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data.datasets import FaceDataset
from src.data.transforms import build_facenet_eval_transform, build_vit_eval_transform
from src.inference.retrieval import build_reference_embeddings, predict_with_late_fusion
from src.models.facenet import FaceNetModel
from src.models.vit import ViTEmbeddingModel


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
	try:
		import yaml
	except ImportError as exc:
		raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from exc

	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Invalid YAML structure in {path}")
	return data


def _resolve_path(value: str, *, base: Path) -> Path:
	value = str(value).strip().strip('"').strip("'")
	if value == "":
		return Path("")
	path = Path(value)
	if path.is_absolute():
		return path
	return (base / path).resolve()


def _read_train_csv(path: Path) -> pd.DataFrame:
	# Supported formats:
	# - header with columns: image_path, gt
	# - no header, two columns
	try:
		df = pd.read_csv(path)
	except Exception:
		df = pd.read_csv(path, header=None)

	if {"image_path", "gt"}.issubset(df.columns):
		train_df = df[["image_path", "gt"]].copy()
	elif df.shape[1] >= 2:
		train_df = pd.read_csv(path, header=None, names=["image_path", "gt"], on_bad_lines="skip").copy()
	else:
		raise ValueError(f"{path} must have at least 2 columns (image_path, gt)")

	train_df = train_df.dropna()
	train_df["image_path"] = train_df["image_path"].astype(str).str.strip()
	train_df["gt"] = train_df["gt"].astype(str).str.strip()
	return train_df


def _read_test_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if "image_path" in df.columns:
		test_df = df[["image_path"]].copy()
	elif df.shape[1] >= 1:
		col = df.columns[0]
		test_df = df[[col]].copy()
		test_df.columns = ["image_path"]
	else:
		raise ValueError(f"{path} must have at least 1 column containing image paths")

	test_df["image_path"] = test_df["image_path"].astype(str).str.strip()
	return test_df


def _make_full_paths(
	df: pd.DataFrame,
	*,
	images_dir: Path,
	fallback_dir: Optional[Path] = None,
) -> pd.DataFrame:
	if "full_path" in df.columns:
		out = df.copy()
		out["full_path"] = out["full_path"].astype(str).str.strip()
		return out

	images_dir_str = str(images_dir) if str(images_dir) else ""
	fallback_dir_str = str(fallback_dir) if fallback_dir is not None and str(fallback_dir) else ""

	def to_full_path(p: str) -> str:
		p = str(p).strip()
		if os.path.isabs(p):
			return p
		if images_dir_str:
			cand = os.path.join(images_dir_str, p)
			if os.path.exists(cand):
				return cand
			return cand
		if fallback_dir_str:
			cand = os.path.join(fallback_dir_str, p)
			return cand
		return p

	out = df.copy()
	out["full_path"] = out["image_path"].apply(to_full_path)
	return out


def _quick_path_sanity_check(df: pd.DataFrame, *, name: str) -> None:
	if len(df) == 0:
		raise ValueError(f"{name} dataframe is empty")
	if "full_path" not in df.columns:
		raise ValueError(f"{name} dataframe is missing full_path")

	sample = df["full_path"].sample(n=min(100, len(df)), random_state=0)
	missing = [p for p in sample.tolist() if not os.path.exists(str(p))]
	if len(missing) == len(sample):
		raise FileNotFoundError(
			f"None of the sampled {name} image paths exist on disk. "
			"Set inference.base_dir/train_images_dir/test_images_dir in the YAML, "
			"or store absolute paths in the CSV. Example missing path: "
			+ str(missing[0])
		)


def _load_facenet_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> None:
	ckpt = torch.load(str(checkpoint_path), map_location=device)
	if isinstance(ckpt, dict):
		state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
	else:
		state = ckpt
	model.load_state_dict(state)


def _load_vit_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> None:
	ckpt = torch.load(str(checkpoint_path), map_location=device)
	if isinstance(ckpt, dict):
		state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
	else:
		state = ckpt

	model_dict = model.state_dict()
	filtered_state_dict = {k: v for k, v in state.items() if k in model_dict}
	model.load_state_dict(filtered_state_dict, strict=False)


def _score_against_gt(*, predictions_df: pd.DataFrame, gt_csv: Path) -> Optional[float]:
	if not gt_csv or str(gt_csv) == "" or not gt_csv.exists():
		return None

	gt_df = pd.read_csv(gt_csv)
	# Supported gt formats:
	# 1) Direct mapping columns: image, gt (or image_path, gt)
	# 2) Tracking-style: columns include 'objects' with python-literal strings like
	#    {'gt': 'person_93', 'image': 'test_set/9198.jpg'} or a list of such dicts.
	if {"image", "gt"}.issubset(gt_df.columns):
		gt_map = gt_df[["image", "gt"]].copy()
	elif {"image_path", "gt"}.issubset(gt_df.columns):
		gt_map = gt_df[["image_path", "gt"]].copy()
		gt_map.columns = ["image", "gt"]
	elif "objects" in gt_df.columns:
		import ast

		records = []
		for raw in gt_df["objects"].astype(str).tolist():
			raw = raw.strip()
			if raw in {"", "[]", "nan", "None"}:
				continue
			try:
				parsed = ast.literal_eval(raw)
			except Exception:
				continue

			items = parsed if isinstance(parsed, list) else [parsed]
			for item in items:
				if not isinstance(item, dict):
					continue
				img = item.get("image") or item.get("image_path")
				gt = item.get("gt")
				if img is None or gt is None:
					continue
				records.append({"image": os.path.basename(str(img)), "gt": str(gt).strip()})

		if not records:
			raise ValueError(
				f"{gt_csv} has an 'objects' column but no parseable image/gt pairs were found"
			)
		gt_map = pd.DataFrame.from_records(records)
	else:
		raise ValueError(
			f"Unsupported gt.csv format: expected columns (image,gt) or (image_path,gt) or an 'objects' column"
		)

	gt_map["image"] = gt_map["image"].astype(str).apply(os.path.basename)
	gt_map["gt"] = gt_map["gt"].astype(str).str.strip()
	gt_map = gt_map.dropna().drop_duplicates(subset=["image"], keep="last")

	p = predictions_df.copy()
	p["image"] = p["image"].astype(str).apply(os.path.basename)
	p["gt"] = p["gt"].astype(str).str.strip()

	merged = p.merge(gt_map, on="image", how="inner", suffixes=("_pred", "_true"))
	if len(merged) == 0:
		raise ValueError("No overlapping image names between predictions and gt.csv")

	acc = (merged["gt_pred"] == merged["gt_true"]).mean()
	return float(acc)


def run_inference_from_config(
	*,
	config_path: Path,
	threshold_override: Optional[float] = None,
	facenet_weight_override: Optional[float] = None,
) -> Tuple[pd.DataFrame, Optional[float]]:
	config_path = config_path.resolve()
	config_dir = config_path.parent

	config = _load_yaml(config_path)
	if "inference" not in config or not isinstance(config["inference"], dict):
		raise ValueError("Config must contain a top-level 'inference' mapping")

	inf = config["inference"]
	repo = _repo_root()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	facenet_checkpoint = _resolve_path(inf["facenet_checkpoint"], base=repo)
	vit_checkpoint = _resolve_path(inf["vit_checkpoint"], base=repo)
	train_csv = _resolve_path(inf["train_csv"], base=repo)
	test_csv = _resolve_path(inf["test_csv"], base=repo)
	gt_csv = _resolve_path(inf.get("gt_csv", ""), base=repo)
	output_csv = _resolve_path(inf["output_csv"], base=repo)

	batch_size = int(inf.get("batch_size", 512))
	num_workers = int(inf.get("num_workers", 2))
	embedding_dim = int(inf.get("embedding_dim", 1024))
	facenet_scaling_factor = float(inf.get("facenet_scaling_factor", 500.0))

	image_sizes = inf.get("image_sizes", {})
	facenet_size = tuple(image_sizes.get("facenet", [160, 160]))
	vit_size = tuple(image_sizes.get("vit", [224, 224]))

	threshold = float(threshold_override if threshold_override is not None else inf.get("threshold", 0.435))
	facenet_weight = float(
		facenet_weight_override if facenet_weight_override is not None else inf.get("facenet_weight", 0.69)
	)

	base_dir = str(inf.get("base_dir", "")).strip().strip('"').strip("'")
	train_images_dir = str(inf.get("train_images_dir", "")).strip().strip('"').strip("'")
	test_images_dir = str(inf.get("test_images_dir", "")).strip().strip('"').strip("'")

	def _is_effectively_empty_path(p: Path) -> bool:
		# Path("") becomes '.', which we should treat as "not provided".
		return str(p).strip() in {"", "."}

	if train_images_dir:
		train_images_root = Path(train_images_dir)
	elif base_dir:
		train_images_root = Path(base_dir)
	else:
		train_images_root = Path("")
	if _is_effectively_empty_path(train_images_root):
		train_images_root = train_csv.parent

	if test_images_dir:
		test_images_root = Path(test_images_dir)
	elif base_dir:
		test_images_root = Path(base_dir) / "test"
	else:
		test_images_root = Path("")
	if _is_effectively_empty_path(test_images_root):
		test_images_root = test_csv.parent

	print(f"Device: {device}")
	print(f"FaceNet checkpoint: {facenet_checkpoint}")
	print(f"ViT checkpoint: {vit_checkpoint}")
	print(f"Train CSV: {train_csv}")
	print(f"Test CSV: {test_csv}")
	if str(gt_csv):
		print(f"GT CSV: {gt_csv}")
	print(f"Train images root: {train_images_root}")
	print(f"Test images root: {test_images_root}")

	# Load models
	facenet_model = FaceNetModel(embedding_dim=embedding_dim, scaling_factor=facenet_scaling_factor).to(device)
	_load_facenet_checkpoint(facenet_model, facenet_checkpoint, device=device)
	print("Loaded FaceNet model")

	vit_model = ViTEmbeddingModel(embedding_dim=embedding_dim, dropout=0.5).to(device)
	_load_vit_checkpoint(vit_model, vit_checkpoint, device=device)
	print("Loaded ViT model")

	# Load dataframes
	print("Loading CSVs...")
	train_df = _read_train_csv(train_csv)
	test_df = _read_test_csv(test_csv)

	train_df = _make_full_paths(train_df, images_dir=train_images_root)
	test_df = _make_full_paths(test_df, images_dir=test_images_root, fallback_dir=(Path(base_dir) if base_dir else None))

	le = LabelEncoder()
	train_df["label"] = le.fit_transform(train_df["gt"])

	_quick_path_sanity_check(train_df, name="train")
	_quick_path_sanity_check(test_df, name="test")
	print(f"Training samples: {len(train_df)} | identities: {len(le.classes_)}")
	print(f"Test samples: {len(test_df)}")

	# Build datasets/loaders (separate transforms per model)
	facenet_transform = build_facenet_eval_transform(tuple(int(x) for x in facenet_size))
	vit_transform = build_vit_eval_transform(image_size=int(vit_size[0]))

	train_dataset_facenet = FaceDataset(train_df, transform=facenet_transform, is_train=True)
	train_dataset_vit = FaceDataset(train_df, transform=vit_transform, is_train=True)
	test_dataset_facenet = FaceDataset(test_df, transform=facenet_transform, is_train=False)
	test_dataset_vit = FaceDataset(test_df, transform=vit_transform, is_train=False)

	pin_memory = device.type == "cuda"
	train_loader_facenet = DataLoader(
		train_dataset_facenet,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	train_loader_vit = DataLoader(
		train_dataset_vit,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	test_loader_facenet = DataLoader(
		test_dataset_facenet,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	test_loader_vit = DataLoader(
		test_dataset_vit,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	print("Generating reference embeddings...")
	ref_f, ref_v = build_reference_embeddings(
		facenet_model=facenet_model,
		vit_model=vit_model,
		train_loader_facenet=train_loader_facenet,
		train_loader_vit=train_loader_vit,
		label_encoder=le,
		device=device,
	)
	print(f"Reference embeddings generated for {len(ref_f)} identities")

	print("Predicting with late fusion...")
	predictions = predict_with_late_fusion(
		facenet_model=facenet_model,
		vit_model=vit_model,
		ref_embeddings_facenet=ref_f,
		ref_embeddings_vit=ref_v,
		test_loader_facenet=test_loader_facenet,
		test_loader_vit=test_loader_vit,
		test_full_paths=test_df["full_path"].tolist(),
		device=device,
		facenet_weight=facenet_weight,
		threshold=threshold,
	)

	predictions_df = pd.DataFrame(predictions)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	predictions_df.to_csv(output_csv, index=False)
	print(f"Saved predictions to {output_csv}")

	acc = _score_against_gt(predictions_df=predictions_df, gt_csv=gt_csv)
	if acc is not None:
		print(f"Accuracy vs gt.csv: {acc:.4f}")
	return predictions_df, acc


def run_inference(*, threshold: float = 0.435, facenet_weight: float = 0.69):
	"""Backward-compatible entrypoint (previously notebook-derived).

	Uses the default config at configs/inference/ensemble.yaml.
	"""

	default_config = _repo_root() / "configs" / "inference" / "ensemble.yaml"
	pred_df, _ = run_inference_from_config(
		config_path=default_config,
		threshold_override=threshold,
		facenet_weight_override=facenet_weight,
	)
	return pred_df.to_dict(orient="records")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run late-fusion inference (FaceNet + ViT) from a YAML config")
	parser.add_argument(
		"--config",
		type=str,
		default=str(_repo_root() / "configs" / "inference" / "ensemble.yaml"),
		help="Path to YAML config (default: configs/inference/ensemble.yaml)",
	)
	parser.add_argument("--threshold", type=float, default=None, help="Override threshold from config")
	parser.add_argument("--facenet-weight", type=float, default=None, help="Override facenet_weight from config")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_inference_from_config(
		config_path=Path(args.config),
		threshold_override=args.threshold,
		facenet_weight_override=args.facenet_weight,
	)

