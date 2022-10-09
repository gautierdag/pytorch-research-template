import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy


class MyProjectModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(512, 10)
        self.val_acc = Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss)
        self.val_acc(y_hat, y)

    def validation_epoch_end(self, outputs):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
