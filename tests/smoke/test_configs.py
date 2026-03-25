"""Smoke tests: verify all config files load and have expected structure."""
import os
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
CONFIG_FILES = list(CONFIGS_DIR.rglob("*.py"))

# Top-level variables expected in generation configs
EXPECTED_VARS = [
    "max_steps",
    "batch_size",
    "num_gpus",
    "model",
    "data",
    "trainer",
]


def _load_config_as_module(config_path: Path) -> dict:
    """Load a Python config file as a plain namespace dict."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_test_config", str(config_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_loads_without_error(config_path):
    """Each config file should be importable as a Python module."""
    cfg = _load_config_as_module(config_path)
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_has_expected_keys(config_path):
    """Each generation config should define the minimum set of variables."""
    cfg = _load_config_as_module(config_path)
    missing = [k for k in EXPECTED_VARS if k not in cfg]
    assert not missing, f"Config {config_path.name} is missing keys: {missing}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_max_steps_positive(config_path):
    cfg = _load_config_as_module(config_path)
    assert cfg["max_steps"] > 0


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_batch_size_positive(config_path):
    cfg = _load_config_as_module(config_path)
    assert cfg["batch_size"] > 0


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_model_has_vae_and_ddm(config_path):
    """model config must contain both 'vae' and 'ddm' sub-configs."""
    cfg = _load_config_as_module(config_path)
    model = cfg.get("model", {})
    assert "vae" in model, f"{config_path.name}: model config missing 'vae'"
    assert "ddm" in model, f"{config_path.name}: model config missing 'ddm'"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_vae_model_has_type(config_path):
    """model.vae.model must have a 'type' key."""
    cfg = _load_config_as_module(config_path)
    vae_model = cfg.get("model", {}).get("vae", {}).get("model", {})
    assert "type" in vae_model, f"{config_path.name}: model.vae.model missing 'type'"
    assert vae_model["type"] == "GraspCVAE"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=[c.name for c in CONFIG_FILES])
def test_config_ddm_model_has_type(config_path):
    """model.ddm.model must have a 'type' key."""
    cfg = _load_config_as_module(config_path)
    ddm_model = cfg.get("model", {}).get("ddm", {}).get("model", {})
    assert "type" in ddm_model, f"{config_path.name}: model.ddm.model missing 'type'"
    assert ddm_model["type"] == "GraspLatentDDM"
