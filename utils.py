

"""
Most of the utility functions here are adapted from 
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO

and modified for our purposes.
"""

import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from torchvision.utils import draw_bounding_boxes


def intersection_over_union(boxes_preds, boxes_labels, box_format="yolo"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    if box_format == "yolo":
        box1_x1 = boxes_preds[..., 0:1] - (boxes_preds[..., 2:3] ** 2) / 2
        box1_y1 = boxes_preds[..., 1:2] - (boxes_preds[..., 3:4] ** 2) / 2
        box1_x2 = boxes_preds[..., 0:1] + (boxes_preds[..., 2:3] ** 2) / 2
        box1_y2 = boxes_preds[..., 1:2] + (boxes_preds[..., 3:4] ** 2) / 2
        box2_x1 = boxes_labels[..., 0:1] - (boxes_labels[..., 2:3] ** 2) / 2
        box2_y1 = boxes_labels[..., 1:2] - (boxes_labels[..., 3:4] ** 2) / 2
        box2_x2 = boxes_labels[..., 0:1] + (boxes_labels[..., 2:3] ** 2) / 2
        box2_y2 = boxes_labels[..., 1:2] + (boxes_labels[..., 3:4] ** 2) / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    #
    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    # print(f"intersection {intersection}")
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=np.random.random()*5+1,
            edgecolor=cm.get_cmap('spring')(np.random.random()),
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def convert_cellboxes(predictions, S=7, B=2, C=2):
    """
    This function is for output visualization purposes and is not needed for training.
    This function is called by cellboxes_to_boxes.
    Do not call this function directly.

    This function takes in output predictions from the model.
    predictions is of shape [batch_size, S*S*(C + 5*B)]

    The return is a tensor of shape
    [batch_size, S, S, 6]

    This function performs the following:
    - (1) for each set of B bounding boxes, pick the one with the highest confidence score
      This turns our matrix from [S,S,B] to [S,S,1]
      and we discard half the predicted boxes
    - (2) reformat our chosen boxes so that each becomes a tensor of length 6
       the tensor is formatted (class_label, confidence, x, y, w, h) 
       Note that the model predicted sqrt(w) and sqrt(h) so during this step,
       we square the predictions to get back w and h
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 5*B)
    bboxes1 = predictions[..., (C+1):(C+5)]
    bboxes2 = predictions[..., (C+6):(C+10)]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., (C+5)].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., 0:1] +  cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] +  cell_indices.permute(0, 2, 1, 3))
    w_y = (best_boxes[..., 2:4] ** 2)
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., 0:C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(
        -1
    ) 
    
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    """
    This function accepts the model output of shape (batch, S*S*(C+2B))
    and returns a list of list of tensors of shpae (batch, S*S, 6)
    It pretty much just does what convert_cellboxes does but also 
    converts everything from tensors into python lists.
    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, 6)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        # print(len(bboxes))
        all_bboxes.append(bboxes)

    return all_bboxes

def baseline_cellboxes_to_boxes(out, stride=64, clf_dim=64*2, img_dim=448):
    """
    Accepts baseline model output of shape (batch, S, S, num_classes)
    and returns list of bounding boxes
    """
    boxes = []
    for b in range(out.shape[0]):
        box = []
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                if(out[b,i,j,0] > 0.95 or out[b,i,j,1] > 0.95):
                    box.append([
                        0,0,(clf_dim/2+stride*i)/448, (clf_dim/2+stride*j)/448, clf_dim/448, clf_dim/448
                    ])
        boxes.append(box)
    return boxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def showData(data, boxes):
    labels_map = {
        0: "Square",
        1: "Circle",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img, labels = data[i]
        c, h, w = img.shape
        figure.add_subplot(rows, cols, i)
        # plt.title(label[:,0])
        boxes = []
        class_labels = []
        for box in labels:
            class_id = box[0]
            class_labels.append(labels_map[int(class_id)])
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            xmin = box[0] - box[2] / 2
            ymin = box[1] - box[3] / 2
            xmax = xmin + box[2]
            ymax = ymax + box[3]
            xmin *= h
            ymin *= w
            xmax *= h
            ymax *= w
            boxes.append(torch.tensor([xmin,ymin,xmax,ymax], dtype=torch.long))
            # Add the patch to the Axes
        img = draw_bounding_boxes(img, boxes, class_labels)
        plt.axis("off")
        plt.imshow(img.numpy().transpose(1,2,0))
    plt.show()