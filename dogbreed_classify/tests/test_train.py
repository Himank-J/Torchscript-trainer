import pytest
import hydra
from pathlib import Path
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import train function
from src.train import train

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=dogbreed_exp", "+trainer.fast_dev_run=True"],
        )
        return cfg

def test_dogbreed_ex_training(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Instantiate components
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)

    # Run training
    train(config, trainer, model, datamodule)

    # Check if training metrics are returned
    assert isinstance(trainer.callback_metrics, dict)  # Ensure metrics are returned
    assert len(trainer.callback_metrics) > 0  # Ensure metrics are not empty
    assert "train/loss" in trainer.callback_metrics  # Check if training loss is recorded
    assert "train/acc" in trainer.callback_metrics  # Check if training accuracy is recorded
    assert trainer.callback_metrics["train/loss"] > 0  # Ensure training loss is greater than 0
    assert trainer.callback_metrics["train/acc"] >= 0 and trainer.callback_metrics["train/acc"] <= 1  # Ensure accuracy is between 0 and 1

    # Check if the training process has started
    assert trainer.current_epoch > 0  # Ensure that the current epoch is greater than 0

    # Test the model's output shape
    sample_input = next(iter(datamodule.train_dataloader()))[0]  # Get a sample input from the dataloader
    output = model(sample_input)  # Forward pass
    assert output.shape[1] == model.hparams.num_classes  # Ensure output shape matches number of classes

    # Run testing
    test_metrics = trainer.test(model, datamodule=datamodule)

    # Check if test metrics are returned
    assert isinstance(test_metrics, list)  # Ensure test metrics are returned as a list
    assert len(test_metrics) > 0  # Ensure test metrics are not empty
    assert "test/loss" in test_metrics[0]  # Check if test loss is recorded
    assert "test/acc" in test_metrics[0]  # Check if test accuracy is recorded
    assert test_metrics[0]["test/loss"] >= 0  # Ensure test loss is non-negative
    assert test_metrics[0]["test/acc"] >= 0 and test_metrics[0]["test/acc"] <= 1  # Ensure accuracy is between 0 and 1
