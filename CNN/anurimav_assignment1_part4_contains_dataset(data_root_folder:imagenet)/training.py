from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from train_test_split import preprocess_dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchvision.datasets import ImageFolder

def train(model, iterator, optimizer, criterion):
    loss = 0
    acc = 0
    model.train()
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        accuracy = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        acc += accuracy
    return loss.detach().numpy() / len(iterator), acc / len(iterator)

def evaluate(model, iterator, criterion, test_set):
    loss = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            loss += loss.item()
            acc += acc
    if test_set== 1:
        return loss / len(iterator), acc / len(iterator), y_pred, y
    else:
        return loss / len(iterator), acc / len(iterator)


def calculate_accuracy(y_pred, y):
    _, topk_predictions = torch.topk(y_pred, 2)
    correct = 0
    for i in range(y.shape[0]):
        if y[i] in topk_predictions[i]:
            correct += 1
    accuracy = correct / y.shape[0]
    return accuracy