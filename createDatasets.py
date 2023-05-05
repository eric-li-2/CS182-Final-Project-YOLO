"""
Written by Eric Li on 4/30/23
Run this python script to create a CircleSquare dataset in the same directory.
The file structure is 
data 
    img
        circle_0.png
        circle_1.png
        ...
    annotations
        labels_0.pt
        labels_1.pt
        ...
The annotations are saved in .pt files and are loaded using pytorch's load function 
which is just a wrapper for pickle

If no data directory exists, this script will create the relevant directories and generate data
If the data directory already exists, this script will overwrite the data in the directory if overwrite is set to True
The overwrite tag is below this comment

If overwrite is set to False, the program exits

There isn't really that much reason to have an overwrite tag but it is just to make 
sure the user actually wants to overwrite data. If you only plan on running this script once,
the tag is irrelevant.
"""
overwrite = True 


import torch
import os
import sys
import torchvision
import matplotlib.pyplot as plt
from torchvision.io import write_png
from tqdm import tqdm 

def intersection(bbox1, bbox2):
    """
    bbox is a 2x2 tensor where the first dimension specifies the point
    and the second dimension specifices xy coord
    The first point is top left and second is bottom right
    """
    left =  max(bbox1[0,0], bbox2[0,0]) 
    right = min(bbox1[1,0], bbox2[1,0])
    top =   max(bbox1[0,1], bbox2[0,1])
    bot =   min(bbox1[1,1], bbox2[1,1])
    if left >= right or top >= bot:
        return 0.0
    return (right - left) * (bot - top)

def bothVisible(bbox1, bbox2, thresh=0.7):
    """
    Returns true if at least thresh proportion of both boxes is visible
    bbox is a tensor of shape (2,2)
    """
    inter = intersection(bbox1, bbox2)
    area1 = (bbox1[1,0] - bbox1[0,0]) * (bbox1[1,1] - bbox1[0,1])
    area2 = (bbox2[1,0] - bbox2[0,0]) * (bbox2[1,1] - bbox2[0,1])
    return inter / area1 < (1 - thresh) and inter / area2 < (1 - thresh)

def generateSingleCircleSquaresImage(height, width, num_classes=3):
    """
    Creates image with a single square or circle or nothing
    """
    # generate a noisy background
    mean = torch.zeros((3, height, width))
    std = torch.ones(mean.shape)
    img = torch.normal(mean, std).abs().clamp(0,1) / 2
    label = None

    # randomly place shape
    rand = torch.rand(1).item()
    center = torch.zeros((2,)).uniform_(0.2,0.8)
    d = torch.zeros((1,)).uniform_(0.1,0.5) # diameter of the circle
    r = d / 2
    x1 = center[0] - r
    y1 = center[1] - r
    x2 = center[0] + r
    y2 = center[1] + r
    cx = center[0] 
    cy = center[1] 

    if rand < (1/3): # draw circle
        x = torch.arange(height, dtype=torch.float).repeat(width,1).permute(1,0)
        y = torch.arange(width, dtype=torch.float).repeat(height,1)
        x -= cx * height 
        y -= cy * width 
        dist = x*x + y*y 
        r_pixel = r * min(width, height)
        r2_pixel = r_pixel * r_pixel
        img += torch.where(dist < r2_pixel, 0.5, 0)
        label = 0

    elif rand < (2/3): # draw square
        x = torch.arange(height, dtype=torch.float).repeat(width,1).permute(1,0)
        y = torch.arange(width, dtype=torch.float).repeat(height,1)
        x -= cx * height 
        y -= cy * width 
        x = torch.abs(x)
        y = torch.abs(y)
        dist = torch.max(torch.stack((x,y), dim=0), dim=0)[0]
        r_pixel = r * min(width, height)
        r2_pixel = r_pixel * r_pixel
        img += torch.where(dist < r_pixel, 0.5, 0)
        label = 1
     
    else: 
        label = 2

    img = img.clamp(0,1)
    img *= 255
    img = img.type(torch.uint8)

    return img, label
    

def generateCircleSquaresImage(height, width, max_circles, thresh=0.8):
    """
    This function returns an image and list of bounding boxes
    The image is a torch.uint8 tensor of shape (3,height, width)
    Each box in the list of bounding boxes is specified by its top left and bottom right corner
    and is a tensor of length 4
    uint8 datatype is used here to write to png easily.
    """
    # generate a noisy background
    mean = torch.zeros((3, height, width))
    std = torch.ones(mean.shape)
    img = torch.normal(mean, std).abs().clamp(0,1) / 2
    bboxes = []
    class_labels = []
    colors = torch.tensor([[1,0,0],
              [0,1,0],
              [0,0,1],
              [0.7, 0.2, 0],
              [0.1, 0.2, 0.7]])
    # generate up to 5 circles in one image
    # similar to NMS algorithm
    # sequentially add squares/circles if they aren't blocked by previously
    # included objects
    for k in range(max_circles):
        center = torch.zeros((2,)).uniform_(0.2,0.8)
        d = torch.zeros((1,)).uniform_(0.1,0.5) # diameter of the circle
        r = d / 2
        x1 = center[0] - r
        y1 = center[1] - r
        x2 = center[0] + r
        y2 = center[1] + r
        cx = center[0] 
        cy = center[1] 
        label = int(torch.zeros(1).uniform_(0,1) > 0.5)

        bbox = torch.tensor([[x1,y1],[x2,y2]])
        if len(bboxes) == 0 or all(bothVisible(bbox, other, thresh) for other in bboxes):
            class_labels.append(torch.tensor([label]))
            bboxes.append(bbox)
            if label == 0: # draw a rectangle
                x = torch.arange(height, dtype=torch.float).repeat(width,1).permute(1,0)
                y = torch.arange(width, dtype=torch.float).repeat(height,1)
                x -= cx * height 
                y -= cy * width 
                x = torch.abs(x)
                y = torch.abs(y)
                dist = torch.max(torch.stack((x,y), dim=0), dim=0)[0]
                r_pixel = r * min(width, height)
                r2_pixel = r_pixel * r_pixel
                img += torch.where(dist < r_pixel, 0.5, 0)
                # for i in range(height):
                #     for j in range(width):
                #         if abs(i/height - cx) <= r and abs(j/width - cy) <= r:
                #             img[:, i, j] += torch.normal(torch.tensor([0.5,0.5,0.5]),
                #                                     torch.tensor([0.1,0.1,0.1])).abs().clamp(0.3,1)
            elif label == 1: # draw a circle
                x = torch.arange(height, dtype=torch.float).repeat(width,1).permute(1,0)
                y = torch.arange(width, dtype=torch.float).repeat(height,1)
                x -= cx * height 
                y -= cy * width 
                dist = x*x + y*y 
                r_pixel = r * min(width, height)
                r2_pixel = r_pixel * r_pixel
                img += torch.where(dist < r2_pixel, 0.5, 0)
                # for i in range(height):
                #     for j in range(width):
                #         if (i/height-cx)**2 + (j/width-cy)**2 < r*r:
                #             img[:, i, j] += torch.normal(torch.tensor([0.5,0.5,0.5]),
                #                                     torch.tensor([0.1,0.1,0.1])).abs().clamp(0.3,1)
    bboxes = [box.flatten() for box in bboxes]
    labels = [torch.cat((label,bbox), dim=0) for label, bbox in zip(class_labels, bboxes)]
    img = img.clamp(0,1)
    img *= 255
    img = img.type(torch.uint8)
    return img, labels

def previewDataset(dataloader, title):
    x_preview, _  = next(iter(dataloader))
    grid = torchvision.utils.make_grid(x_preview, nrow=x_preview.shape[0]).permute(1,2,0)
    plt.figure(figsize=[20, 20])
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    plt.show()

def download_multiobj_dataset():
    overwrite = True
    img_root = os.path.join(".", "data", "img") 
    annotations_root = os.path.join(".", "data", "annotations")
    try:
        os.makedirs(img_root)
        os.makedirs(annotations_root)
    except FileExistsError:
        print("Data files already exist")
        if not overwrite:
            sys.exit("Overwite set to false, Set overwite to true or delete data directory")
        print("Overwritting existing circle data files")

    num_images = 400
    height = 448
    width = 448

    for i in tqdm(range(num_images)):
        img, labels = generateCircleSquaresImage(height, width, 5)
        img_path = os.path.join(img_root, f"circle_{i}.png")
        label_path = os.path.join(annotations_root,  f"labels_{i}.pt")
        write_png(img, img_path)
        torch.save(labels, label_path)

def download_clf_dataset():
    overwrite = True
    img_root = os.path.join(".", "clf_data", "img") 
    annotations_root = os.path.join(".", "clf_data", "annotations")
    try:
        os.makedirs(img_root)
        os.makedirs(annotations_root)
    except FileExistsError:
        print("Data files already exist")
        if not overwrite:
            sys.exit("Overwite set to false, Set overwite to true or delete data directory")
        print("Overwritting existing circle data files")

    num_images = 400
    height = 128
    width = 128

    for i in tqdm(range(num_images)):
        img, labels = generateSingleCircleSquaresImage(height, width)
        img_path = os.path.join(img_root, f"circle_{i}.png")
        label_path = os.path.join(annotations_root,  f"labels_{i}.pt")
        write_png(img, img_path)
        torch.save(labels, label_path)

def main():
    # create data directories
    img_root = os.path.join(".", "data", "img") 
    annotations_root = os.path.join(".", "data", "annotations")
    try:
        os.makedirs(img_root)
        os.makedirs(annotations_root)
    except FileExistsError:
        print("Data files already exist")
        if not overwrite:
            sys.exit("Overwite set to false, Set overwite to true or delete data directory")
        print("Overwritting existing circle data files")

    num_images = 400

    for i in tqdm(range(num_images)):
        img, labels = generateSingleCircleSquaresImage(128, 128)
        img_path = os.path.join(img_root, f"circle_{i}.png")
        label_path = os.path.join(annotations_root,  f"labels_{i}.pt")
        write_png(img, img_path)
        torch.save(labels, label_path)

if __name__ == "__main__":
    main()