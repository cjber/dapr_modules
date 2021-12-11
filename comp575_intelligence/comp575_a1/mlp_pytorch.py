import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = self.ce(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    pl.seed_everything(42)
    dataset = CIFAR10(root="data/", transform=transforms.ToTensor(), download=True)
    mlp = MLP()
    trainer = pl.Trainer(
        logger=None, gpus=-1, auto_select_gpus=True, deterministic=False, max_epochs=5
    )
    trainer.fit(mlp, DataLoader(dataset, batch_size=128, num_workers=16))
