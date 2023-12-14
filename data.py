from torch.utils.data import Dataset

class TeacherDataset(Dataset):
    def __init__(self, X, y_t) -> None:
        super().__init__()
        self.X = X
        self.y_t = y_t

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y_t[index]
