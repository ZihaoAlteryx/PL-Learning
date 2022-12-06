#https://www.kaggle.com/code/bootiu/dog-vs-cat-transfer-learning-by-pytorch-lightning
import numpy as np
import pandas as pd
import os, glob, time, copy, random, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Data Augmentation
class ImageTransform():
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        # Transformimg Image
        img_transformed = self.transform(img, self.phase)
        
        # Get Label
        if "F1" in img_path:
            label = 1
        elif "Car" in img_path:
            label = 0

        return img_transformed, label

class vgg16_model(pl.LightningModule):
    
    def __init__(self, img_path, criterion, batch_size, img_size):
        super(vgg16_model, self).__init__()
        self.criterion = criterion
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Load Data  ###############################################################################
        self.img_path = img_path
        # Split Train/Val Data
        self.train_img_path = self.img_path
        self.val_img_path = self.img_path
        # Dataset
        self.train_dataset = CustomDataset(self.train_img_path, 
                                             ImageTransform(self.img_size), 
                                             phase='train')
        
        self.val_dataset = CustomDataset(self.val_img_path, 
                                           ImageTransform(self.img_size), 
                                           phase='val')
        
        # Model  ###############################################################################
        # Pretrained VGG16
        use_pretrained = True
        self.net = models.vgg16(pretrained=use_pretrained)
        # Change Output Size of Last FC Layer (4096 -> 1)
        self.net.classifier[6] = nn.Linear(in_features=self.net.classifier[6].in_features, out_features=2)
        # Specify The Layers for updating
        params_to_update = []
        update_params_name = ['classifier.6.weight', 'classifier.6.bias']

        for name, param in self.net.named_parameters():
            if name in update_params_name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        # Set Optimizer
        self.optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    
    # Method  ###############################################################################
    # Set Train Dataloader

    def train_dataloader(self):
        '''
        REQUIRED
        Set Train Dataloader
        '''
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    
    # Set Valid Dataloader

    def val_dataloader(self):
        '''
        REQUIRED
        Set Validation Dataloader
        '''
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)
    
    def forward(self, x):
        return self.net(x)
    
    # Set optimizer, and schedular
    def configure_optimizers(self):
        # [optimizer], [schedular]
        return [self.optimizer], []
    
    # Train Loop
    def training_step(self, batch, batch_idx):
        '''
        REQUIRED
        batch: Output from DataLoader
        batch_idx: Index of Batch
        '''
        
        # Output from Dataloader
        imgs, labels = batch
        
        # Prediction (try remove the forward and test)
        preds = self.forward(imgs)
        # Calc Loss
        loss = self.criterion(preds, labels)
        
        # Calc Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)
        
        logs = {'train_loss': loss, 'train_correct': correct}
        
        return {'loss': loss, 'log': logs, 'progress_bar': logs}
    
    # Validation Loop
    def validation_step(self, batch, batch_idx):
        '''
        OPTIONAL
        SAME AS "trainning_step"
        '''
        # Output from Dataloader
        imgs, labels = batch
        
        # Prediction
        preds = self.forward(imgs)
        # Calc Loss
        loss = self.criterion(preds, labels)
        
        # Calc Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)
        
        logs = {'val_loss': loss, 'val_correct': correct}
        
        return {'val_loss': loss, 'val_correct': correct, 'log': logs, 'progress_bar': logs}
    
    # Aggegate Validation Result
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_correct': avg_correct}
        torch.cuda.empty_cache()

        return {'avg_val_loss': avg_loss, 'log': logs}
# Config  ################################################
criterion = nn.CrossEntropyLoss()
batch_size = 16
img_size = 224
epoch = 10

# Set LightningSystem  ################################################
train_img_path = ["data/F1_1.jpg","data/F1_2.jpg","data/F1_3.jpg","data/F1_4.jpg","data/F1_5.jpg","data/F1_6.jpg","data/F1_7.jpg","data/F1_8.jpg","data/F1_9.jpg", "data/F1_10.jpg"
                    ,"data/Car_1.jpg","data/Car_2.jpg","data/Car_3.jpg","data/Car_4.jpg","data/Car_5.jpg","data/Car_6.jpg"]
model = vgg16_model(train_img_path, criterion, batch_size, img_size)

# Callbacks  ################################################
# Save Model
# checkpoint_callback = ModelCheckpoint(filepath='SAVE_FILE_PATH', monitor='val_loss',
#                                       save_best_only=True, mode='min', save_weights_only=True)
# EarlyStopping
# earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)

# Trainer  ################################################
trainer = Trainer(
    max_epochs=epoch,                        # Set Num Epoch
#     default_save_path=output_path,            # Path for save lightning_logs
#     checkpoint_callback=checkpoint_callback,  # Set Checkpoint-Callback
#     early_stop_callback=earlystopping,        # Set EarlyStopping-Callback
    gpus=[0]                                  # GPU
)

# Start Training!!  ################################################
trainer.fit(model)

