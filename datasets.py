import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import *
from math import sqrt
import os 
from createDatasets import generateCircleSquaresImage, generateSingleCircleSquaresImage

class CircleSquareYOLODataset(Dataset):
    def __init__(self, data_dir=None, S=7, B=2, C=2):
        """
        This dataset returns a tuple (image, label_matrix)
        where image is of shape [3,448, 448]
        and label_matrix is of shape [S,S,C+2*B]

        It takes an input a path to an existing data directory 
        created by running createDatasets.py 
        or generates on the spot, (image, label) pairs.
        If genearting images during runtime,
        note that querying the dataset by index will 
        not give you the same image generated previously for that index.

        This class is mainly a wrapper class that an image and list of bounding boxes 
        and creates a label matrix out of the bounding boxes

        The label matrix generation code is adapted from Aladdin Person's implementation
        https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
        """

        self.S = S
        self.B = B
        self.C = C
        if data_dir == None: 
            # generate images as they are queried
            # mainly used for test/validation test set
            self.num_images = 200
            self.img_dir = None
            self.ann_dir = None
            return
        self.img_dir = os.path.join(data_dir, "img")
        self.ann_dir = os.path.join(data_dir, "annotations")
        

        if not os.path.exists(self.img_dir):
           print("image directory missing")
        if not os.path.exists(self.ann_dir):
            print("annotations directory missing") 
        self.num_images = len(os.listdir(self.img_dir))
        len1 = len(os.listdir(self.img_dir))
        len2 = len(os.listdir(self.ann_dir))
        assert len1 == len2, f"The image directory and annotation directory have unequal file amounts {len1} != {len2}"

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """
        This function returns an image, label_matrix pair
        The label matrix is generated inside this function as it is dependent
        on our choice of S, B, and C, so we chose not 
        to store our data labels with label matrices precomputed.
        """
        assert 0 <= idx  and idx < self.num_images
        if self.img_dir == None:
            img, labels = generateCircleSquaresImage(448, 448, 5)
        else:
            img_path = os.path.join(self.img_dir, f"circle_{idx}.png")
            img = read_image(img_path)
            ann_path = os.path.join(self.ann_dir, f"labels_{idx}.pt")
            labels = torch.load(ann_path)
        # return img, labels 

        # img is shape [3,H,W] and takes values [0...255]
        # labels is a list of tensor(5) objects which are the bounding boxes
        # the first element is the class id and the next four are x1, y1, x2, y2
        # return img, labels
        # turn labels and bboxes into label matrix
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # TODO

        for box in labels:
            class_label, x1, y1, x2, y2 = box.tolist()
            class_label = int(class_label)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            h = x2 - x1
            w = y2 - y1

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
       
            w = sqrt(w) # predict sqrt(width) relative to whole image [0...1]
            h = sqrt(h)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, w, h]
                )

                label_matrix[i, j, (self.C+1):(self.C+5)] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
        
        img = img.float() / 255
        return img, label_matrix


class CircleSquareClassifierDataset(Dataset):
    def __init__(self, data_dir=None, height=64*4, width=64*4):
        """
        This dataset returns a tuple (image, label_matrix)
        where image is of shape [3,448, 448]
        and label_matrix is of shape [S,S,C+2*B]
        """

        self.height = height
        self.width = width

        if data_dir == None: 
            # generate images as they are queried
            # mainly used for test/validation test set
            self.num_images = 200
            self.img_dir = None
            self.ann_dir = None
            return
        self.img_dir = os.path.join(data_dir, "img")
        self.ann_dir = os.path.join(data_dir, "annotations")

        if not os.path.exists(self.img_dir):
           print("image directory missing")
        if not os.path.exists(self.ann_dir):
            print("annotations directory missing") 
        self.num_images = len(os.listdir(self.img_dir))
        len1 = len(os.listdir(self.img_dir))
        len2 = len(os.listdir(self.ann_dir))
        assert len1 == len2, f"The image directory and annotation directory have unequal file amounts {len1} != {len2}"

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """
        This function returns an image, label pair
        """
        assert 0 <= idx  and idx < self.num_images
        if self.img_dir == None:
            img, labels = generateSingleCircleSquaresImage(self.height, self.width, 5)
        else:
            img_path = os.path.join(self.img_dir, f"circle_{idx}.png")
            img = read_image(img_path)
            ann_path = os.path.join(self.ann_dir, f"labels_{idx}.pt")
            labels = torch.load(ann_path)
        
        img = img.float() / 255
        return img, labels