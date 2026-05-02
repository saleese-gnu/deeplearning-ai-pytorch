# cifar10_dataset.py
from torch.utils.data import Dataset

class CIFAR10HFWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label