'''
equivalent implementation of exer_two using pytorch_lightning
* the defined methods are distinctive for PyTorch Lightning
* data loader approach is via the LightningDataModule 
(most organized among the three approaches)
'''

# general
import pytorch_lightning as pl 
import torch

# model definition
import torch.nn as  nn

from torchmetrics import Accuracy

# setting up data loaders
from torch.utils.data import DataLoader
from torch.utils.data import random_split
 
from torchvision.datasets import MNIST
from torchvision import transforms

# training
from pytorch_lightning.callbacks import ModelCheckpoint

# seed
torch.manual_seed(1) 

# Defining the Model
class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        
        # new PL attributes:
        self.train_acc = Accuracy(task = 'multiclass', num_classes = 10)
        self.valid_acc = Accuracy(task = 'multiclass', num_classes = 10)
        self.test_acc = Accuracy(task = 'multiclass', num_classes = 10)
        
        # Model similar to previous section:
        input_size = image_shape[0] * image_shape[1] * image_shape[2] 
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 10)) 
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Setting up Data Loaders
class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def prepare_data(self):
        MNIST(root=self.data_path, download=True) 

    def setup(self, stage=None):
        # stage is either 'fit', 'validate', 'test', or 'predict'
        # here note relevant
        mnist_all = MNIST( 
            root=self.data_path,
            train=True,
            transform=self.transform,  
            download=False
        ) 

        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST( 
            root=self.data_path,
            train=False,
            transform=self.transform,  
            download=False
        ) 

    # num_workers=0 due to Runtime errors 
    # - need to look out for this to increase the number of workers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=0)

# data
mnist_dm = MnistDataModule()

# model
mnistclassifier = MultiLayerPerceptron()

callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")] # save top 1 model

# Trainer class
# - takes care of all the intermediate steps such as calling backward(), zero_grad(), optimizer.step()
# - lets us easily specify the number of gpus to use

'''
# training from epoch 0
if torch.cuda.is_available(): # if you have GPUs
    trainer = pl.Trainer(   max_epochs=10, 
                            callbacks=callbacks, 
                            accelerator = 'gpu', devices = 1)
else:
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)
'''

# continuing training from the model checkpoint
if torch.cuda.is_available(): # if you have GPUs
    trainer = pl.Trainer(   max_epochs=15, 
                            callbacks=callbacks, # added part to get still model checkpoint 
                            accelerator = 'gpu', devices = 1,
                            resume_from_checkpoint = 'C:\\Users\\ASUS\\Desktop\\self_study\\github\\pytorch-playground\\chap13\\5_pytorch_lightning\\lightning_logs\\version_6\\checkpoints\\epoch=13-step=12040.ckpt')
else:
    trainer = pl.Trainer(   max_epochs=10, 
                            callbacks=callbacks,
                            resume_from_checkpoint = 'C:\\Users\\ASUS\\Desktop\\self_study\\github\\pytorch-playground\\chap13\\5_pytorch_lightning\\lightning_logs\\version_6\\checkpoints\\epoch=13-step=12040.ckpt')


trainer.fit(model=mnistclassifier, datamodule=mnist_dm)

# test model
trainer.test(model=mnistclassifier, datamodule=mnist_dm)
