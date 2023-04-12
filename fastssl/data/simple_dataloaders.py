import numpy as np
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self,
                 data_labels_dict,
                 X_key='activations',
                 y_key='labels'):
        self.X = data_labels_dict[X_key]
        self.y = data_labels_dict[y_key]

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    
def SimpleDataloader(fname,
                     splits=['train','test'],
                     batch_size=512,
                     num_workers=2):
    data_from_file = np.load(fname,allow_pickle=True).item()
    loaders = {}
    for split in splits:
        dataset = SimpleDataset(data_labels_dict=data_from_file[split])
        loaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    return loaders
