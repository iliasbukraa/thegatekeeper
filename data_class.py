from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from PIL import Image
from torch import tensor, long

class MaskDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return \
            {'image': self.transformations(Image.imread(row['image'])),
             'mask': tensor(row['mask'], dtype=long),
             }
