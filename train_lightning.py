import math
import os

import cv2
import open_clip
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import yaml
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import linalg
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup


class config:
    name = "vit_224_v3"
    root_dir = r"/home/nick/Data"
    lr_model = 1e-6
    lr_fc = 1e-4
    weight_decay = 1e-5
    epochs = 25
    batch_size = 64
    img_size = 224
    scheduler = 'cos' # could be 'cos' or 'step'
    warmup_epochs = 1
    num_workers = 12
    num_classes = 9691
    embedding_size = 768
    precision = 16


def get_train_aug():
    train_augs = tv.transforms.Compose(
        [
            tv.transforms.RandomResizedCrop((config.img_size, config.img_size)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_augs


def get_val_aug():
    val_augs = tv.transforms.Compose(
        [
            tv.transforms.Resize((config.img_size, config.img_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return val_augs


def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    return img


class Product10KDataset(data.Dataset):
    def __init__(
        self, root, annotation_file, transforms, is_inference=False, with_bbox=False
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist["img_path"][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)


def get_dataloaders():
    print("Preparing train reader...")
    train_dataset = Product10KDataset(
        root=os.path.join(config.root_dir, "train"),
        annotation_file=os.path.join(config.root_dir, "train.csv"),
        transforms=get_train_aug(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print("Done.")

    print("Preparing valid reader...")
    val_dataset = Product10KDataset(
        root=os.path.join(config.root_dir, "test"),
        annotation_file=os.path.join(config.root_dir, "test.csv"),
        transforms=get_val_aug(),
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    print("Done.")

    return train_loader, valid_loader, train_dataset


class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=30, m=0.3):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)
        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos**2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
        return cos


class Classifier_model(nn.Module):
    def __init__(self):
        super(Classifier_model, self).__init__()
        self.model = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )[0].visual
        self.fc = ArcFace(config.embedding_size, config.num_classes)

    def forward(self, x, labels=None):
        x = self.model(x)
        x = self.fc(x, labels)
        return x


class VPRModule(pl.LightningModule):
    def __init__(self, num_train_steps):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        # self.save_hyperparameters()
        # Create model
        self.model = Classifier_model()
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.num_train_steps = num_train_steps
        # Example input for visualizing the graph in Tensorboard

    #         self.sample_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    #         self.example_input_array = [self.sample_input, self.sample_input]

    def forward(self, img, labels):
        # Forward function that is run when visualizing the graph
        return self.model(img, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.model.parameters(), "lr": config.lr_model},
                {"params": self.model.fc.parameters(), "lr": config.lr_fc},
            ],
            weight_decay=config.weight_decay,
        )
        if config.scheduler == 'cos':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=self.num_train_steps * config.warmup_epochs,
                                                         num_training_steps=int(
                                                             self.num_train_steps * config.epochs))
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[config.epochs-5, config.epochs-1], gamma=0.1
            )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        img, labels = batch
        preds = self.model(img, labels)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        preds = self.model(img)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)


def main():
    config_dict = config.__dict__
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}
    os.makedirs(os.path.join(config.root_dir, "model_saves", config.name))
    with open(
        os.path.join(config.root_dir, "model_saves", config.name, "config.yaml"), "w"
    ) as file:
        yaml.dump(config_dict, file)

    train_loader, val_loader, train_dataset = get_dataloaders()
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.root_dir, "model_saves", config.name),
        mode="max",
        every_n_epochs=1,
        monitor="val_acc",
        save_last=True,
        save_weights_only=True,
    )
    model = VPRModule(len(train_dataset) // config.batch_size)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        precision=config.precision,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(config.root_dir, "model_saves", config.name)
        ),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
