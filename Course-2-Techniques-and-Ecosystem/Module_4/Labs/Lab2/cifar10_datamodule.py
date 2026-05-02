# cifar10_datamodule.py
import lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from cifar10_dataset import CIFAR10HFWrapper  # your wrapper from earlier

class CIFAR10HFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        train_raw = load_dataset("uoft-cs/cifar10", split="train")
        val_raw   = load_dataset("uoft-cs/cifar10", split="test")
        self.train_ds = CIFAR10HFWrapper(train_raw, transform=self.transform)
        self.val_ds   = CIFAR10HFWrapper(val_raw,   transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)