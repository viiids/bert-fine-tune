from torch.utils.data import DataLoader

# Creating instances of training and validation set
from bert.SSTDataset import SSTDataset


def load_data():
    train_set = SSTDataset(filename='resources/train.tsv', maxlen=30)
    val_set = SSTDataset(filename='resources/dev.tsv', maxlen=30)

    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=64, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=5)

    return train_loader, val_loader
