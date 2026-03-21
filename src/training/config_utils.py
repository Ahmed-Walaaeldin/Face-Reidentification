from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
	path = Path(path)
	try:
		import yaml
	except ImportError as exc:
		raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from exc

	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Invalid YAML structure in {path}")
	return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	"""Merge two nested dicts (override wins)."""
	out: Dict[str, Any] = deepcopy(base)
	_stack: list[tuple[MutableMapping[str, Any], Mapping[str, Any]]] = [(out, override)]
	while _stack:
		current_out, current_override = _stack.pop()
		for key, value in current_override.items():
			if isinstance(value, dict) and isinstance(current_out.get(key), dict):
				_stack.append((current_out[key], value))
			else:
				current_out[key] = deepcopy(value)
	return out


def _get_by_dot_path(root: Mapping[str, Any], path: str) -> Any:
	cur: Any = root
	for part in path.split("."):
		if not isinstance(cur, Mapping) or part not in cur:
			raise KeyError(path)
		cur = cur[part]
	return cur


def _interpolate_string(value: str, *, root: Mapping[str, Any]) -> str:
	def repl(match: re.Match[str]) -> str:
		expr = match.group(1).strip()
		try:
			resolved = _get_by_dot_path(root, expr)
		except KeyError:
			# Leave unresolved placeholders as-is.
			return match.group(0)
		return str(resolved)

	return _INTERP_RE.sub(repl, value)


def interpolate_config(config: Dict[str, Any], *, max_passes: int = 10) -> Dict[str, Any]:
	"""Resolve ${a.b.c} placeholders inside string values."""
	out = deepcopy(config)
	for _ in range(max_passes):
		changed = False

		def walk(obj: Any) -> Any:
			nonlocal changed
			if isinstance(obj, dict):
				return {k: walk(v) for k, v in obj.items()}
			if isinstance(obj, list):
				return [walk(v) for v in obj]
			if isinstance(obj, str) and "${" in obj:
				new_val = _interpolate_string(obj, root=out)
				if new_val != obj:
					changed = True
				return new_val
			return obj

		out = walk(out)
		if not changed:
			break
	return out


def resolve_path(value: str | Path, *, base: Optional[Path] = None) -> Path:
	"""Resolve a path-like value relative to base (defaults to repo root)."""
	if base is None:
		base = repo_root()
	if isinstance(value, Path):
		p = value
	else:
		v = str(value).strip().strip('"').strip("'")
		if v == "":
			return Path("")
		p = Path(v)
	if p.is_absolute():
		return p
	return (base / p).resolve()


def is_windows() -> bool:
	return os.name == "nt"
