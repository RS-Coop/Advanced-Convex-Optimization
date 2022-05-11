'''
Various custom dataloaders for SARC experiments
'''

'''
Dataset class for spambase dataset
'''
class SpamBase(Dataset):
    def __init__(self, dataroot, split, device, data_map=None):
        assert split in ['train', 'test']

        self.data = pd.read_csv(f'{dataroot}/spambase_{split}.data', header=None)

        if data_map is None:
            self.data_map = lambda x: np.log(x+0.1)
        else:
            self.data_map = data_map

        self.device = device

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.data.iloc[idx, 0:-1].apply(self.data_map),
                            dtype=torch.float, device=self.device)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float,
                            device=self.device).reshape(-1)

        return {'features':X, 'labels':y}
