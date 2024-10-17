# resnet.py
import multiprocessing
import os
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import datetime
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional.classification import f1_score, accuracy
from torchmetrics.classification import MultilabelAveragePrecision
from torchvision.models import resnet50, ResNet50_Weights

if torch.__version__ >= '2.0':
    torch.set_float32_matmul_precision('high')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yaml')
train_config = config['train']
data_config = config['data']
logging_config = config['logging']
set_seed(train_config['seed'])

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 경로 가져오기
        image_path = self.data.iloc[idx, 0]

        # 레이블 가져오기
        labels = self.data.iloc[idx, 1:].values.astype(np.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 이미지 로드
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 이미지 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, labels


transform = transforms.Compose([
    transforms.Resize(data_config['transform']['resize']),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_config['transform']['mean'], std=data_config['transform']['std'])
])

train_csv = './split_data/train.csv'
val_csv = './split_data/val.csv'
test_csv = './split_data/test.csv'

train_dataset = CustomDatasetFromCSV(train_csv, transform=transform)
val_dataset = CustomDatasetFromCSV(val_csv, transform=transform)
test_dataset = CustomDatasetFromCSV(test_csv, transform=transform)

num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=num_workers)

# 모델 정의
class ResNetLightningModel(LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(ResNetLightningModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.train_mAP = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        self.val_mAP = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        self.test_mAP = MultilabelAveragePrecision(num_labels=num_classes, average='micro')

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels.float())

        preds_prob = torch.sigmoid(outputs)

        self.train_mAP.update(preds_prob, labels)

        f1 = f1_score(preds_prob, labels, task="multilabel",
                    num_labels=self.num_classes, average='weighted')
        # acc = accuracy(preds_prob, labels, task="multilabel",
        #             num_labels=self.num_classes, average='micro', threshold=0.5)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_f1', f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels.float())

        preds_prob = torch.sigmoid(outputs)

        self.val_mAP.update(preds_prob, labels)

        f1 = f1_score(preds_prob, labels, task="multilabel",
                    num_labels=self.num_classes, average='weighted')
        # acc = accuracy(preds_prob, labels, task="multilabel",
        #             num_labels=self.num_classes, average='micro', threshold=0.5)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels.float())

        preds_prob = torch.sigmoid(outputs)

        self.test_mAP.update(preds_prob, labels)

        f1 = f1_score(preds_prob, labels, task="multilabel",
                    num_labels=self.num_classes, average='weighted')
        # acc = accuracy(preds_prob, labels, task="multilabel",
        #             num_labels=self.num_classes, average='micro', threshold=0.5)

        # self.log('test_loss', loss, sync_dist=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        train_mAP_value = self.train_mAP.compute()
        self.log('train_mAP', train_mAP_value, prog_bar=True, sync_dist=True)
        self.train_mAP.reset()

    def on_validation_epoch_end(self):
        val_mAP_value = self.val_mAP.compute()
        self.log('val_mAP', val_mAP_value, prog_bar=True, sync_dist=True)
        self.val_mAP.reset()

    def on_test_epoch_end(self):
        test_mAP_value = self.test_mAP.compute()
        self.log('test_mAP', test_mAP_value, prog_bar=True, sync_dist=True)
        self.test_mAP.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def on_train_end(self):
        save_dir = logging_config['local_dirpath']
        model_name = f"resnet_final_epoch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        save_path = os.path.join(save_dir, model_name)
        self.save_model(save_path)
        print(f"Model saved at: {save_path}")

num_classes = data_config['num_classes']
learning_rate = train_config['learning_rate']

model = ResNetLightningModel(num_classes=num_classes, learning_rate=learning_rate)

# 로깅 및 체크포인트 설정
now = datetime.datetime.now().strftime("%m%d_%H%M")
wandb_logger = WandbLogger(project=logging_config['project_name'], log_model=logging_config['log_model'], name=f"resnet_{now}")

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=os.path.join(logging_config['local_dirpath'], 'resnet'),
    filename='resnet-{val_loss:.2f}-{val_f1:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    max_epochs=train_config['max_epochs'],
    deterministic=False,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    accelerator='gpu',
    devices=1
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

wandb.finish()
