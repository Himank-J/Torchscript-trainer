import os
from pathlib import Path
import logging
import json

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper
from src.models.dogbreed_classifier import DogBreedClassifier
from src.datamodules.dogbreed_datamodule import DogBreedDataModule

log = logging.getLogger(__name__)


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@task_wrapper
def evaluate(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting evaluation!")
    results = trainer.test(model, datamodule=datamodule)
    
    # Save results as JSON
    results_file = Path(cfg.paths.output_dir) / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results[0], f, indent=2)
    
    log.info(f"Evaluation results saved to {results_file}")
    log.info(f"Evaluation metrics:\n{results}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    output_dir = Path(cfg.paths.output_dir)

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Load the model from checkpoint
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(cfg.ckpt_path)

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
        default_root_dir=str(output_dir),
    )

    # Evaluate the model
    evaluate(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()