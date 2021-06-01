import os
import random
import time
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, add_hist
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

import dataset 
import model 

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from torch.optim.lr_scheduler import StepLR

plt.rcParams['axes.grid'] = False

def seed_everything(random_seed):
    # seed 고정
    # random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def train(num_epochs, model, data_loader, val_loader, criterion, criterion1, optimizer, saved_dir, val_every, device, scheduler):
    print('Start training..')
    best_miou = 0
    for epoch in range(num_epochs):
        hist = np.zeros((12, 12))
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
                        
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            loss += criterion1(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            # step 주기에 따른 loss, mIoU 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item(), mIoU))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_miou = validation(epoch + 1, model, val_loader, criterion, criterion1, device)
            if val_miou > best_miou:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_miou = val_miou
                save_model(model, saved_dir)
        scheduler.step()

def validation(epoch, model, data_loader, criterion, criterion1, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((12, 12))
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss += criterion1(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mIoU))

    return avrg_loss, mIoU

def main():
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
    print('GPU 이름: {}' .format(torch.cuda.get_device_name(0)))
    print('GPU 개수" {}' .format(torch.cuda.device_count()))

    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

    batch_size = 16   # Mini-batch size
    num_epochs = 60
    learning_rate = 0.0001


    # 모델 저장 함수 정의
    val_every = 1 

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
        
    def save_model(model, saved_dir, file_name='deeplabv3plus_baseline.pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)

    # Loss function 정의
    # criterion = nn.CrossEntropyLoss()
    criterion = smp.losses.DiceLoss('multiclass', eps = 1e-7)
    criterion1 = smp.losses.FocalLoss('multiclass', alpha = 0.5, gamma = 2.0, reduction = 'mean')

    # Optimizer 정의
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    scheduler = StepLR(optimizer, 15, gamma=0.1)

    train(num_epochs, model, train_loader, val_loader, criterion, criterion1, optimizer, saved_dir, val_every, device, scheduler)

if __name__ == "__main__":

    main()