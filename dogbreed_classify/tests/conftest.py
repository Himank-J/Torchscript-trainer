import pytest
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import rootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@pytest.fixture(scope="session", autouse=True)
def set_project_root():
    os.environ["PROJECT_ROOT"] = str(root)

@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train")
        # If you have defaults you want to override, you can do it here
        # cfg = compose(config_name="train", overrides=["data.batch_size=32"])
    return cfg

@pytest.fixture(scope="function")
def hydra_cfg():
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train")
    return cfg

@pytest.fixture(scope="function")
def datamodule(hydra_cfg):
    return hydra.utils.instantiate(hydra_cfg.data)

@pytest.fixture(scope="function")
def model(hydra_cfg):
    return hydra.utils.instantiate(hydra_cfg.model)