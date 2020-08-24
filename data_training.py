'''
This script defines the model architecture of the mask detection model we aim to create. By using the pytorch libraries
we created the MaskDetect class which is the model architecture and further more some boilerplate code every Pytorch model
needs such as the dataloader, optimizer etc. Last step is the the actual training of the model using pytorch. In this case
the epoch with the best accuracy is saved.
'''

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn.init as init
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_class import MaskClass

# Define the model architecture and its layers
class MaskDetect(pl.LightningModule):
    def __init__(self, maskPath:Path=None):
        super(MaskDetect, self).__init__()
        self.maskPath = maskPath

        self.mask_dataframe = None
        self.train_dataframe = None
        self.crossEntropyLoss = None

        self.convolution1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.convolution2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.convolution3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2),
        )

        for sequence in [self.convolution1, self.convolution2, self.convolution3, self.linear]:
            for layer in sequence.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    init.xavier_uniform_(layer.weight)

# define the forward function
    def forward(self, x: Tensor):
        out = self.convolution1(x)
        out = self.convolution2(out)
        out = self.convolution3(out)
        out = out.view(-1, 2048)
        out = self.linear(out)
        return out

# define the function calling the data, the trainloader, the training step and the optimizer in this training step
    def prepare_data(self):
        self.mask_dataframe = pd.read_pickle(self.maskPath)
        self.train_dataframe = MaskClass(self.mask_dataframe)

        mask_pictures = self.mask_dataframe[self.mask_dataframe['mask'] == 1].shape[0]
        nonmask_pictures = self.mask_dataframe[self.mask_dataframe['mask'] == 0].shape[0]
        number_samples = [nonmask_pictures, mask_pictures]
        normalized = [1 - (x / sum(number_samples)) for x in number_samples]
        self.crossEntropyLoss = nn.CrossEntropyLoss(weight=torch.tensor(normalized))

    def train_dataloader(self):
        return DataLoader(self.train_dataframe, batch_size=32, shuffle=True, num_workers=4)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-5)

    def training_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        return {'loss': loss}

# trains the mask detection model using pytorch lightning
if __name__ == '__main__':
    model = MaskDetect(Path('mask_df.pickle'))

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=10,
                      profiler=True)
    trainer.fit(model)
