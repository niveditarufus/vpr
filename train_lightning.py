import math
import os
from collections import OrderedDict
from copy import deepcopy

import albumentations as A
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
from timm import optim
from torch import linalg
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup
from vit_pytorch.distill import DistillableViT, DistillWrapper

# from open_clip_280_overlap.src import open_clip


class config:
    name = "vit_224_v40"
    root_dir = r"/home/nick/Data/"  # could be '/home/nick/Data/' or '/mnt/data/nick/'
    csv_file_tr = "train_balance.csv"  # could be 'train.csv' or 'final_data_224/final_data_224.csv'
    csv_file_tt = (
        "test.csv"  # could be 'test.csv' or 'final_data_224/final_data_224.csv'
    )
    dataset = "Product10KDataset"  # could be 'BigDataset' or 'Product10KDataset'

    num_models_save = 15
    lr_model = 2e-7
    lr_fc = 2e-4
    weight_decay = 1e-2
    epochs = 15
    warmup_epochs = 1
    # start_ema_epoch = 5
    model_freeze_epochs = 0
    teacher_model_path = ""  # "/home/nick/Data/model_saves/epoch=7-step=17736.ckpt"
    start_model_path = ""
    augmentation = True
    color = [
        0.2,
        0.3,
        0.1,
        0.1,
        0.3,
    ]  # brightness=.2, hue=.3, contrast=0, saturation=0, prob=0.5
    dynamic_margin = True
    m = 0.3
    s = 30
    stride = 0.05
    max_m = 0.8
    batch_size = 32
    img_size = 224
    neck = ""  # could be '' or 'option-D'
    scheduler = "cos"  # could be 'cos' or 'step'
    model_name = (
        "ViT-H-14"  # could be 'ViT-H-14' or 'ViT-bigG-14' or 'convnext_xxlarge'
    )
    num_workers = 12
    num_classes = 9691  # could be '14087' or '9691'
    embedding_size = 1024  # 1024  # 1280 or 1024
    proj = True
    precision = 16
    use_val = False
    optimizer = "AdamW"  # could be 'lion' or 'AdamW', 'AdamW_st_tc'
    loss = "CE"  # could be 'CE' or 'LabelSmoothing'


def get_train_aug():
    if config.augmentation:
        train_augs = tv.transforms.Compose(
            [
                tv.transforms.Resize((config.img_size, config.img_size)),
                # tv.transforms.RandomResizedCrop((config.img_size, config.img_size), scale=(0.6, 1.0)),
                # tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomHorizontalFlip(),
                # tv.transforms.RandomApply(
                #    [tv.transforms.RandomRotation(degrees=90)], p=0.3
                # ),
                tv.transforms.RandomApply(
                    [
                        tv.transforms.ColorJitter(
                            brightness=config.color[0],
                            hue=config.color[1],
                            contrast=config.color[2],
                            saturation=config.color[3],
                        ),
                    ],
                    p=config.color[-1],
                ),
                # tv.transforms.RandomApply([tv.transforms.RandAugment()], p=0.3),
                tv.transforms.ToTensor(),
                # tv.transforms.RandomErasing(p=0.3, scale=(0.05, 0.3), ratio=(0.3, 3.3)),
                # tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                tv.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    else:
        train_augs = tv.transforms.Compose(
            [
                tv.transforms.Resize((config.img_size, config.img_size)),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
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


class BigDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference

    def __getitem__(self, index):
        cv2.setNumThreads(6)
        if self.is_inference:
            impath, _ = self.imlist.iloc[index]
        else:
            impath, target = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


def get_dataloaders():
    print("Preparing train reader...")
    if config.dataset == "BigDataset":
        train_dataset = BigDataset(
            root=os.path.join(config.root_dir),
            annotation_file=os.path.join(config.root_dir, config.csv_file_tr),
            transforms=get_train_aug(),
        )
    elif config.dataset == "Product10KDataset":
        train_dataset = Product10KDataset(
            root=os.path.join(config.root_dir, "train"),
            annotation_file=os.path.join(config.root_dir, config.csv_file_tr),
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
    if config.dataset == "BigDataset":
        val_dataset = BigDataset(
            root=os.path.join(config.root_dir),
            annotation_file=os.path.join(config.root_dir, config.csv_file_tt),
            transforms=get_val_aug(),
        )
    elif config.dataset == "Product10KDataset":
        val_dataset = Product10KDataset(
            root=os.path.join(config.root_dir, "test"),
            annotation_file=os.path.join(config.root_dir, config.csv_file_tt),
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
    def __init__(self, cin, cout, s=30, m=0.3, stride=0.1, max_m=0.8):
        super().__init__()
        self.m = m
        self.s = s
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.cos_m = torch.cos(torch.tensor(self.m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)
        self.last_epoch = 0
        self.max_m = max_m
        self.m_s = stride

    def update(self, c_epoch):
        self.m = min(self.m + self.m_s * (c_epoch - self.last_epoch), self.max_m)
        self.last_epoch = c_epoch
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.cos_m = torch.cos(torch.tensor(self.m))

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


class Neck(nn.Module):
    def __init__(self, in_features, out_features, style="high_dim"):
        super().__init__()
        if style == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                torch.nn.PReLU(),
            )
        elif style == "simple":
            self.neck = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, out_features),
            )
        elif style == "norm_double_linear":
            self.neck = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.3),
                nn.Linear(in_features, out_features * 2),
                nn.Linear(out_features * 2, out_features),
            )
        elif style == "norm_single_linear":
            self.neck = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.2),
                nn.Linear(in_features, out_features),
            )
        else:
            raise NotImplementedError(f"Unkown Neck: {stype}")

    def forward(self, x):
        return self.neck(x)


class Classifier_model(nn.Module):
    def __init__(self):
        super(Classifier_model, self).__init__()
        if config.model_name == "ViT-H-14":
            pretrained = "laion2b_s32b_b79k"
        elif config.model_name == "ViT-bigG-14":
            pretrained = "laion2b_s39b_b160k"
        if config.model_name == "convnext_xxlarge":
            self.model = open_clip.create_model_and_transforms(
                "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
            )[0].visual
        else:
            self.model = open_clip.create_model_and_transforms(
                config.model_name,
                pretrained=pretrained
                # "ViT-H-14-252", pretrained=pretrained
            )[0].visual
        if not config.proj:
            self.model.proj = None
        if config.neck != "":
            print(
                "Neck size = ",
                self.model.state_dict()["proj"].shape[-1],
                config.embedding_size,
            )
            self.neck = Neck(
                self.model.state_dict()["proj"].shape[-1],
                config.embedding_size,
                "option-D",
            )
        self.fc = ArcFace(
            config.embedding_size,
            config.num_classes,
            s=config.s,
            m=config.m,
            stride=config.stride,
            max_m=config.max_m,
        )

    def forward(self, x, labels=None):
        x = self.model(x)
        if config.neck != "":
            x = self.neck(x)
        x = self.fc(x, labels)
        return x


class ModelEmaV2(torch.nn.Module):
    def __init__(self, model, decay=0.9995, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_student_teacher_distiller():
    teacher_model = Classifier_model()
    state_dict_model = torch.load(config.teacher_model_path)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict_model.items():
        if "model." in k:
            name = k.replace("model.", "", 1)
        new_state_dict[name] = v
    teacher_model.load_state_dict(new_state_dict)
    student_model = DistillableViT(
        image_size=224,
        patch_size=14,
        num_classes=config.num_classes,
        dim=1024,
        depth=32,
        heads=16,
        mlp_dim=5120,
        dropout=0.1,
        emb_dropout=0.1,
    )
    distiller = DistillWrapper(
        student=student_model,
        teacher=teacher_model,
        temperature=3,  # temperature of distillation
        alpha=0.5,  # trade between main loss and distillation loss
        hard=False,  # whether to use soft or hard distillation
    )
    return distiller


class VPRModule(pl.LightningModule):
    def __init__(self, num_train_steps):
        super().__init__()
        if config.teacher_model_path != "":
            self.model = get_student_teacher_distiller()
        else:
            self.model = Classifier_model()

        if config.start_model_path != "":
            try:
                new_state_dict = torch.load(config.start_model_path)["model_state_dict"]
            except:
                state_dict_model = torch.load(config.start_model_path)["state_dict"]
                new_state_dict = OrderedDict()
                new_state_dict_fc = OrderedDict()
                for k, v in state_dict_model.items():
                    if "fc.fc" in k:
                        name = k.replace("model.fc.", "")
                        name = name.replace("fc.fc", "fc")
                        new_state_dict_fc[name] = v
                    else:
                        name = k.replace("model.", "")
                        # name = k.replace("module.backbone.net.", "")
                        new_state_dict[name] = v

            self.model.model.load_state_dict(new_state_dict)
            self.model.fc.load_state_dict(new_state_dict_fc)

        # self.model_ema = ''
        if config.loss == "CE":
            self.loss_module = nn.CrossEntropyLoss()
        elif config.loss == "LabelSmoothing":
            self.loss_module = LabelSmoothingCrossEntropy()
        self.num_train_steps = num_train_steps

    def forward(self, img, labels):
        return self.model(img, labels)

    def configure_optimizers(self):
        if config.optimizer == "lion":
            self.optimizer = optim.lion.Lion(
                [
                    {"params": self.model.model.parameters(), "lr": config.lr_model},
                    {"params": self.model.fc.parameters(), "lr": config.lr_fc},
                ],
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "AdamW":
            params = [
                {"params": self.model.model.parameters(), "lr": config.lr_model},
                {"params": self.model.fc.parameters(), "lr": config.lr_fc},
            ]
            if config.neck != "":
                params.append(
                    {"params": self.model.neck.parameters(), "lr": config.lr_fc}
                )
            self.optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)
        elif config.optimizer == "AdamW_st_tc":
            self.optimizer = torch.optim.AdamW(
                params=self.model.student.parameters(),
                lr=config.lr_fc,
                weight_decay=config.weight_decay,
            )
        if config.scheduler == "cos":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.num_train_steps * config.warmup_epochs),
                num_training_steps=int(self.num_train_steps * config.epochs),
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[config.epochs - 5, config.epochs - 1],
                gamma=0.1,
            )
        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        img, labels = batch

        if config.teacher_model_path == "":  # regular training
            if self.current_epoch < config.model_freeze_epochs:
                for param in self.model.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.model.parameters():
                    param.requires_grad = True

            # if self.current_epoch > -1:
            #    for param in self.model.fc.parameters():
            #        param.requires_grad = False

            preds = self.model(img, labels)
            loss = self.loss_module(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            # Logs the accuracy per epoch to tensorboard (weighted average over batches)
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

            for i, param_group in enumerate(self.optimizer.param_groups):
                self.log(
                    f"lr/lr{i}",
                    param_group["lr"],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

            if config.dynamic_margin:
                self.model.fc.update(self.current_epoch)

        else:  # distillation
            loss = self.model(img, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        if config.scheduler == "cos":
            self.scheduler.step()

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        if config.teacher_model_path != "":
            preds = self.model.student(img)
        else:
            preds = self.model(img)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

    # def optimizer_step(self, epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
    #                   using_native_amp=False, using_lbfgs=False):
    #    optimizer.step(closure=closure)
    #    if epoch >= config.start_ema_epoch:
    #        if self.model_ema == '':
    #            self.model_ema = ModelEmaV2(self.model)
    #        self.model_ema.update(self.model)


def main():
    config_dict = config.__dict__
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}
    os.makedirs(os.path.join(config.root_dir, "model_saves", config.name))
    with open(
        os.path.join(config.root_dir, "model_saves", config.name, "config.yaml"), "w"
    ) as file:
        yaml.dump(config_dict, file)

    train_loader, val_loader, train_dataset = get_dataloaders()

    if config.use_val:
        monitor = "val_acc"
    else:
        monitor = "train_loss"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.root_dir, "model_saves", config.name),
        mode="max",
        save_top_k=config.num_models_save,
        every_n_epochs=1,
        monitor=monitor,
        save_weights_only=True,
    )
    model = VPRModule(num_train_steps=len(train_dataset) // config.batch_size)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        precision=config.precision,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(config.root_dir, "model_saves", config.name)
        ),
    )

    if config.use_val:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
