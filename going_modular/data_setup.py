
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int):
    """Create training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them into
    PyTorch Datasets and then intro PyTorch DataLoaders.
    """

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True)
    test_dataloader = DataLoader(test_data,
                                    batch_size=batch_size,
                                    shuffle=False, 
                                    pin_memory=True)

    return train_dataloader, test_dataloader, class_names
