import pytest
import rootutils
import hydra
from omegaconf import DictConfig
import os

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import the DogBreedDataModule
from src.datamodules.dogbreed_datamodule import DogBreedDataModule

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=dogbreed_exp"]  # Use the appropriate experiment
        )
        return cfg

@pytest.fixture
def datamodule(config):
    # Instantiate the data module with necessary configurations
    return DogBreedDataModule(data_dir=config.paths.data_dir, batch_size=32)

def test_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

def test_datamodule_splits(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    
    assert len(datamodule.train_dataset) / total_size == pytest.approx(0.8, abs=0.01)
    assert len(datamodule.val_dataset) / total_size == pytest.approx(0.1, abs=0.01)
    assert len(datamodule.test_dataset) / total_size == pytest.approx(0.1, abs=0.01)

def test_datamodule_dataloaders(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check that the number of batches is as expected
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

def test_datamodule_batch_size(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    assert batch[0].shape[0] == datamodule.batch_size  # Check batch size

def test_data_augmentation(datamodule):
    # Assuming you have some data augmentation logic
    datamodule.prepare_data()
    datamodule.setup()
    
    # Get a sample from the training dataset
    sample = datamodule.train_dataset[0]
    
    # Check if the sample has been augmented (this will depend on your augmentation logic)
    assert sample is not None  # Replace with actual checks based on your augmentation