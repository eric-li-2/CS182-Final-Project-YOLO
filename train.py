import os
import sys
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim


from datasets import CircleSquareYOLODataset
from model import Yolov1
from loss import YoloLoss
from utils import (
    cellboxes_to_boxes,
    convert_cellboxes,
    non_max_suppression,
    # mean_average_precision,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    showData,
)

# seed = 182
# torch.manual_seed(seed)

def train_fn(train_loader, model, optimizer, loss_fn):
    """
    Trains YOLO model with specified train_loader, optimizer, and loss function
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def testModel(test_loader, model):
    """
    Performs inference on YOLO model and plots bounding boxes on input image
    """
    img, label_matrix = next(iter(test_loader))
    img, label_matrix = img.to(DEVICE), label_matrix.to(DEVICE)
    out = model(img) # Shape [B,S*S*(C+5*B)]
    pred_boxes = cellboxes_to_boxes(out) # shape [B, S*S, 6]
    # true_boxes = cellboxes_to_boxes(label_matrix)[0]
    nms_pred_boxes = non_max_suppression(pred_boxes[0], 0.2, 0.5)
    plot_image(img[0].permute(2,1,0).to("cpu"), nms_pred_boxes)
    # nms_pred_boxes = [non_max_suppression(pred_boxes[i]) for i in range(9)]
    # showData(img, nms_pred_boxes)

if __name__ == '__main__':
    # Hyperparameters etc. 
    LEARNING_RATE = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available else "cpu"
    # DEVICE = "cpu"
    BATCH_SIZE = 8 # 64 in original paper but I don't have that much vram, grad accum?
    WEIGHT_DECAY = 0
    EPOCHS = 10
    # NUM_WORKERS = 2
    # PIN_MEMORY = True
    LOAD_MODEL = False
    LOAD_MODEL_FILE = "pretrained.pth.tar"
    # IMG_DIR = "data/images"
    # LABEL_DIR = "data/labels"

    data_dir = os.path.join('.', 'data')
    train_dataset = CircleSquareYOLODataset(data_dir)
    test_dataset = CircleSquareYOLODataset(None) # generate images during runtime

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        # pin_memory=PIN_MEMORY,
        shuffle=True, # TODO
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        # pin_memory=PIN_MEMORY,
        shuffle=True,
        # drop_last=True,
    )

    model = Yolov1(S=7, B=2, C=2).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=7, B=2, C=2)

    if LOAD_MODEL:
        print("Loading pre-trained model...")
        if os.path.exists(LOAD_MODEL_FILE):
            checkpoint = torch.load(LOAD_MODEL_FILE)
            load_checkpoint(checkpoint, model, optimizer)
        else:
            sys.exit(f"File {LOAD_MODEL_FILE} not found. Exiting")
    else:
        print("Training model...")
        for epoch in range(EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn)
        checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
        print("Saving model...")
        save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        import time
        time.sleep(10)
        sys.exit("done")
    
    testModel(test_loader, model)