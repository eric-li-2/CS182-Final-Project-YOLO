import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets import CircleSquareYOLODataset, CircleSquareClassifierDataset
from model import Yolov1, BaselineClassifier, BaselineObjectDetector
from loss import YoloLoss
from utils import (
    cellboxes_to_boxes,
    convert_cellboxes,
    baseline_cellboxes_to_boxes,
    non_max_suppression,
    # mean_average_precision,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    showData,
)

def train_fn(train_loader, model, optimizer, loss_fn, device):
    """
    Trains YOLO model with specified train_loader, optimizer, and loss function
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def clf_accuracy(test_loader, model, device):
    """
    Returns accuracy of baseline classifier predicting squares vs circles
    """
    model.eval()
    total = len(test_loader)
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            correct += (torch.argmax(y_hat, axis=1) == y).float().mean()
    print('Test accuracy: {:.3f}'.format(correct / total))

def testModel(test_loader, model, device):
    """
    Performs inference on YOLO model and plots bounding boxes on input image
    """
    img, label_matrix = next(iter(test_loader))
    img, label_matrix = img.to(device), label_matrix.to(device)
    out = model(img) # Shape [B,S*S*(C+5*B)]
    pred_boxes = cellboxes_to_boxes(out) # shape [B, S*S, 6]
    nms_pred_boxes = non_max_suppression(pred_boxes[0], 0.2, 0.5)
    plot_image(img[0].permute(2,1,0).to("cpu"), nms_pred_boxes)

def testBaseline(test_loader, model, device):
    """
    Perform inference on baseline object detector and plots bounding boxes
    """
    img, _ = next(iter(test_loader))
    img = img.to(device)[[0], :]
    out = model(img) # Shape [B,S*S*(C+5*B)]
    pred_boxes = baseline_cellboxes_to_boxes(out) # shape [B, S*S, 6]
    plot_image(img[0].permute(2,1,0).to("cpu"), pred_boxes[0])