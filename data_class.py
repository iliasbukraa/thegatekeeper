import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class MaskClass(Dataset):
    def __init__(self, df):
        self.df = df

        self.transform = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, key):
        col = self.df.iloc[key]
        image = cv2.imdecode(np.fromfile(col['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            'image': self.transform(image),
            'mask': tensor([col['mask']], dtype=long),
        }
