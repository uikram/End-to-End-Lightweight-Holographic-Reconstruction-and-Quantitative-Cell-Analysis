"""YAML configuration loading with inheritance and attribute access.

All numeric constants used by this project live in the YAML files under
``config/``; nothing here supplies a default value of its own. A missing key is
therefore a configuration error and is reported as one.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_EXTENDS_KEY = "extends"


class ConfigError(KeyError):
    """Raised when a requested configuration key is absent."""


class Config(Mapping):
    """Read-only nested mapping with attribute and dotted-path access."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: Mapping[str, Any], path: str = ""):
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise ConfigError(f"missing configuration key: {self._qualify(key)}")
        return _wrap(self._data[key], self._qualify(key))

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- convenience ------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except ConfigError as exc:
            raise AttributeError(str(exc)) from None

    def __setattr__(self, key: str, value: Any):
        raise TypeError("Config is read-only; edit the YAML file instead")

    def __repr__(self) -> str:
        return f"Config({self._path or 'root'}: {sorted(self._data)})"

    def _qualify(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``key`` if present, otherwise ``default``.

        Reserved for genuinely optional keys. Required values should be read
        with ``cfg.key`` so a missing entry fails loudly.
        """
        if key not in self._data:
            return default
        return _wrap(self._data[key], self._qualify(key))

    def path(self, dotted: str) -> Any:
        """Look up a nested value, e.g. ``cfg.path("loss.weights.phase")``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Config):
                raise ConfigError(f"cannot descend into '{part}' of '{dotted}'")
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def merged(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with ``overrides`` deep-merged on top."""
        return Config(_deep_merge(self.to_dict(), overrides), self._path)


def _wrap(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return Config(value, path)
    if isinstance(value, list):
        return [_wrap(v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config, resolving a chain of ``extends`` references.

    ``extends`` is interpreted relative to the directory of the file declaring
    it. Child values are deep-merged over parent values.
    """
    path = Path(path).expanduser().resolve()
    raw = _read_yaml(path)

    parent_name = raw.pop(_EXTENDS_KEY, None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        data = _deep_merge(parent.to_dict(), raw)
    else:
        data = raw

    if overrides:
        data = _deep_merge(data, overrides)

    data.setdefault("experiment_name", path.stem)
    data["_config_file"] = str(path)
    return Config(data)


def parse_overrides(items: Iterable[str]) -> dict:
    """Turn ``["training.epochs=5", "data.modality=gabor"]`` into a nested dict."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        dotted, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def save_config(cfg: Config, destination: str | Path) -> Path:
    """Write the fully resolved configuration next to a run's outputs."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, default_flow_style=False)
    return destination
