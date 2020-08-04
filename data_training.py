from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,
                      Sequential)
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_class import MaskClass


class MaskDetect(pl.LightningModule):
    def __init__(self, maskDFPath: Path = None):
        super(MaskDetect, self).__init__()
        self.maskDFPath = maskDFPath

        self.maskDF = None
        self.trainDF = None
        self.validateDF = None
        self.crossEntropyLoss = None
        self.learningRate = 0.00001

        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linearLayers = linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )

        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor):
        """ forward pass
        """
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out

    def prepare_data(self):
        self.maskDF = maskDF = pd.read_pickle(self.maskDFPath)
        self.trainDF = MaskClass(self.maskDF)

        maskNum = maskDF[maskDF['mask'] == 1].shape[0]
        nonMaskNum = maskDF[maskDF['mask'] == 0].shape[0]
        nSamples = [nonMaskNum, maskNum]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.crossEntropyLoss = CrossEntropyLoss(weight=torch.tensor(normedWeights))

    def train_dataloader(self):
        return DataLoader(self.trainDF, batch_size=32, shuffle=True, num_workers=4)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learningRate)

    def training_step(self, batch: dict, _batch_idx: int):
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        return {'loss': loss}

if __name__ == '__main__':
    model = MaskDetect(Path('mask_df.pickle'))

    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=1,
                      default_root_dir='weights.ckpt',
                      profiler=True)
    trainer.fit(model)
