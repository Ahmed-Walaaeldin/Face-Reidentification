from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


def compute_inverse_frequency_sample_weights(labels: Iterable[int]):
	labels_array = np.asarray(list(labels), dtype=np.int64)
	class_counts = np.bincount(labels_array)
	class_weights = 1.0 / (class_counts + 1e-6)
	return class_weights[labels_array]

