from pathlib import Path
from typing import Optional, Tuple
import os

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class DogBreedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        num_workers: int = 0,
        batch_size: int = 32,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_val_test_split = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.test_dataset = None  # Add this line

    def prepare_data(self):
        """Download images if not already downloaded and extracted."""
        if not (self.data_dir / "dogbreed_data").exists():
            download_and_extract_archive(
                url="https://sensitivedatashareblob.blob.core.windows.net/files/dogbreed_data.zip",
                download_root=self.data_dir,
                remove_finished=True,
            )

        # Check if the dataset directory is empty
        if not os.listdir(self.data_dir):
            raise ValueError("Dataset is empty")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            print(self.data_dir)
            full_dataset = ImageFolder(root=self.data_dir / "dogbreed_data" /"train", transform=self.transforms)
            train_size = int(self.train_val_test_split[0] * len(full_dataset))
            val_size = int(self.train_val_test_split[1] * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        else:
            full_dataset = ImageFolder(root=self.data_dir / "dogbreed_data" /"train", transform=self.transforms)
            train_size = int(self.train_val_test_split[0] * len(full_dataset))
            val_size = int(self.train_val_test_split[1] * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,  
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )