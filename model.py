# ライブラリの読み込み
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import torchmetrics
from torchmetrics.functional import accuracy

from PIL import Image

######### 以下アプリケーションには不要。学習時に使用する ########

# デバイスのセットアップ
device = "cuda" if torch.cuda.is_available() else "cpu"

# データの準備
# ファイルの解凍
# 拡張子をtar.gzからtarに変換する
# !mv /content/drive/MyDrive/Colab/kikagaku/自走課題/dataset/food-101.tar.gz /content/drive/MyDrive/Colab/kikagaku/自走課題/dataset/food-101.tar

# # tar.gz ファイルを解凍する
# !tar xvf /content/drive/MyDrive/Colab/kikagaku/自走課題/dataset/food-101.tar

# ラベルの定義
labels = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio",
    "Beef tartare", "Beet salad", "Beignets", "Bibimbap",
    "Bread pudding", "Breakfast burrito", "Bruschetta", "Caesar salad",
    "Cannoli", "Caprese salad", "Carrot cake", "Ceviche",
    "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla",
    "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros",
    "Clam chowder", "Club sandwich", "Crab cakes", "Creme brulee",
    "Croque madame", "Cup cakes", "Deviled eggs", "Donuts",
    "Dumplings", "Edamame", "Eggs benedict", "Escargots",
    "Falafel", "Filet mignon", "Fish and chips", "Foie gras",
    "French fries", "French onion soup", "French toast", "Fried calamari",
    "Fried rice", "Frozen yogurt", "Garlic bread", "Gnocchi",
    "Greek salad", "Grilled cheese sandwich", "Grilled salmon", "Guacamole",
    "Gyoza", "Hamburger", "Hot and sour soup", "Hot dog",
    "Huevos rancheros", "Hummus", "Ice cream", "Lasagna",
    "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons",
    "Miso soup", "Mussels", "Nachos", "Omelette",
    "Onion rings", "Oysters", "Pad thai", "Paella",
    "Pancakes", "Panna cotta", "Peking duck", "Pho",
    "Pizza", "Pork chop", "Poutine", "Prime rib",
    "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake",
    "Risotto", "Samosa", "Sashimi", "Scallops",
    "Seaweed salad", "Shrimp and grits", "Spaghetti bolognese", "Spaghetti carbonara",
    "Spring rolls", "Steak", "Strawberry shortcake", "Sushi",
    "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare",
    "Waffles"
]

# # パスとラベルを読み込み、ラベルの定義
# def load_image_paths_and_labels(txt_file, images_dir):
#     """
#     txtファイルから画像のパスとラベルを読み込み、リストを返す。
#     """
#     with open(txt_file, 'r') as f:
#         lines = f.readlines()

#     image_paths = []
#     labels = []
#     for line in lines:
#         # ファイル名とラベルIDが空白または他の区切り文字で分割されていると仮定
#         parts = line.strip().split('/')
#         image_path = os.path.join(images_dir, parts[0], parts[1] + '.jpg')
#         label = parts[0]  # ラベルはフォルダ名から取得
#         image_paths.append(image_path)
#         labels.append(label)

#     return image_paths, labels

# images_dir = 'food-101/images'  # 画像が格納されているディレクトリ
# train_txt = 'food-101/meta/train.txt'
# test_txt = 'food-101/meta/test.txt'

# # train_image_paths, train_labels = load_image_paths_and_labels(train_txt, images_dir)
# # test_image_paths, test_labels = load_image_paths_and_labels(test_txt, images_dir)

# # データセットのクラスを定義
# class CustomDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform
#         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(labels)))}

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         label = self.class_to_idx[self.labels[idx]]
#         image = Image.open(image_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label
      
# データの前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 水平方向にランダムに反転
    transforms.RandomRotation(10),  # -10度から10度の間でランダムに回転
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色彩の変更
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # データセットの分割
# train_dataset = CustomDataset(train_image_paths, train_labels, transform=transform)
# test_dataset = CustomDataset(test_image_paths, test_labels, transform=transform)

# n_val = int(len(train_dataset) * 0.2)
# n_train = len(train_dataset) - n_val
# pl.seed_everything(42)
# train, val = random_split(train_dataset, [n_train, n_val])

# # バッチサイズの定義
# batch_size = 32

# # データローダーの定義
# train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
# val_loader = DataLoader(val, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ネットワークの定義
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet152

class FoodNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.feature = models.resnet18(pretrained=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, 101)  # Food101データセットは101クラス

    def forward(self, x):
        h = self.feature(x)
        h = self.dropout(h)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return optimizer
      
# # 学習の実行
# num_classes = 101
# pl.seed_everything(42)
# food_model = FoodNet()
# logger = CSVLogger(save_dir='logs', name='my_exp')
# trainer = pl.Trainer(max_epochs=30, accelerator='gpu', deterministic=False, logger=logger)
# trainer.fit(food_model, train_loader, val_loader)

# # モデルの保存
# torch.save(food_model.state_dict(), 'models/food_model.pt')