from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint_state(path: str | Path, *, model_state_dict: Dict[str, Any], **extra: Any) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {"model_state_dict": model_state_dict, **extra}
	torch.save(payload, str(path))


def load_checkpoint_state(path: str | Path, *, map_location=None, weights_only: bool | None = None) -> Dict[str, Any]:
	"""Loads a torch checkpoint.

	The stage-2 FaceNet notebook uses torch.load(..., weights_only=True). That kwarg
	exists in newer torch versions; here we support both.
	"""

	path = Path(path)
	if weights_only is None:
		return torch.load(str(path), map_location=map_location)

	try:
		return torch.load(str(path), map_location=map_location, weights_only=weights_only)
	except TypeError:
		return torch.load(str(path), map_location=map_location)

