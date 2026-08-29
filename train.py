import os
import numpy as np
import torch
import time
import pickle
import pandas as pd
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from sklearn.model_selection import train_test_split

import evaluate

DIRS = []

def set_consts(dirs):
    global DIRS
    DIRS = dirs

class ObjectDetectionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        img = Image.open(img_path)

        boxes = self.data.iloc[idx, 1:5].values.reshape(-1, 4).astype(float)
        labels = torch.tensor([['can', 'bottle', 'container'].index(self.data.iloc[idx, 5])+1], dtype=torch.int64)

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = labels

        # Apply any transformations if specified
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
    
def updateEstimations(past, new):
    prev = past[0]
    for i in range(len(past)):
        (past[i], prev) = (prev, past[i])
    past[0] = new
    sum = 0
    for i in past:
        sum += i
    return sum / len(past)

def do_training(band, bs, test=None, path=None):
    train_data = pd.read_csv(os.path.join(DIRS[0], DIRS[2][0], f"train_{band}.csv"))
    test_path = os.path.join(DIRS[0], DIRS[2][0], f"test_{band}.csv") 
    if test:
        train_data = train_data[100:100+int(len(train_data)*test[0])]
    model_path = path if path else os.path.join(DIRS[0], DIRS[2][0], f"b{bs}_pth", f"model_{band}.pth")
    img_dir = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], f"chips_{band}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ObjectDetectionDataset(train_data, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), pin_memory = True)

    numBatches = len(dataloader)

    # Load the pre-trained SSD model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssdlite320_mobilenet_v3_large(pretrained=True).to(device)
    num_classes = 3
    model.head.classification_head.num_classes = num_classes+1

    # Optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    num_epochs = test[1] if test else 250
    
    at = time.perf_counter()
    bt = time.perf_counter()
    pastEpochEstimations = [0,0,0]
    pastTrainingEstimations = [0,0,0,0,0,0,0,0,0,0]
    best = float('inf')
    p = 0
    prev = float('inf')   

    # Training loop
    print(f"TRAINING STARTED ON {band}")
    for epoch in range(1,num_epochs+1):
        model.train()
        running_loss = 0.0
        currentBatch = 0
        for images, targets in dataloader:
            bt = time.perf_counter()

            batchTime = bt - at
            epochTimeLeft = updateEstimations(pastEpochEstimations, batchTime * (numBatches - currentBatch))
            trainingTimeLeft = updateEstimations(pastTrainingEstimations, epochTimeLeft + (num_epochs - epoch) * batchTime * numBatches)
            print("                                        {Batch " + "{:03}".format(currentBatch) + "/" + str(numBatches) + " - " + "{:09.8f}".format(batchTime) + "s | Epoch " + "{:02}".format(epoch) + "/" + str(num_epochs) + " ETA - 00:" + "{:02}".format(int(epochTimeLeft // 60)) + ":" + "{:02}".format(int(epochTimeLeft - epochTimeLeft // 60 * 60)) + " | Training ETA - " + "{:02}".format(int(trainingTimeLeft // 3600)) + ":" + "{:02}".format(int((trainingTimeLeft - trainingTimeLeft // 3600 * 3600) // 60)) + ":" + "{:02}".format(int(trainingTimeLeft - trainingTimeLeft // 3600 * 3600 - (trainingTimeLeft - trainingTimeLeft // 3600 * 3600) // 60 * 60)) + " | " + "{:08.4f}".format(100 * (currentBatch + numBatches * (epoch - 1)) / (num_epochs * numBatches)) + "%" + "}                  ", end="\r")
            currentBatch += 1

            at = time.perf_counter()
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in t.items()} 
                for t in targets
            ]
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        
        loss = running_loss / len(dataloader)
        """
        if epoch % 3 == 1:
            print("validating...", end = '\r')
            val_loss = validate(model, vd, device, multi_task_loss)
            prev = val_loss
        """
        print(f"Epoch [{epoch}/{num_epochs}], Loss: [{loss:.4f}, {prev:.4f}]")

        if prev >= loss:
            p = 0
        else:
            p += 1
        if p >= 5:
            print(f"\n Training ended early after epoch #{epoch}")
            break

        prev = loss
        scheduler.step(loss)

    torch.save(model.state_dict(), model_path)
    print(f'Training successful     saved model_{band}.pth to folder b{bs}_pth                           ')

def validate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = [img.to(device) for img in images]
            labels = {key: value.to(device) for key, value in labels.items()}

            outputs = model(images)
            loss = 0
            for idx, output in enumerate(outputs):
                loss += criterion(idx, output, labels)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

###################################################################################################
# loss stuff that makes no sense

def multi_task_loss(index, output, targets, giou_weight=1.0, classification_weight=1.0, alpha=0.25, gamma=2.0):
    """
    Multi-task loss function combining Generalized IoU loss for bounding boxes
    and focal loss for classification.

    Args:
        outputs: Model predictions with keys 'boxes' and 'labels'.
        targets: Ground truth data with keys 'boxes' and 'labels'.
        giou_weight: Weight for the Generalized IoU loss.
        classification_weight: Weight for the classification loss.
        alpha: Balancing factor for the focal loss.
        gamma: Focusing parameter for the focal loss.

    Returns:
        total_loss: Combined loss as a scalar tensor.
    """
    giou_losses = []
    classification_losses = []    
    # Extract boxes and labels from predictions and ground truth
    pred_boxes = output['boxes']  # (num_pred_boxes, 4)
    true_boxes = targets['boxes'][index]  # (num_true_boxes, 4)
    pred_labels = output['labels']  # (num_pred_boxes, num_classes)
    true_labels = targets['labels'][index].squeeze(-1)  # (num_true_boxes,)

    # Generalized IoU Loss
    if pred_boxes.shape[0] > 0 and isinstance(true_boxes, torch.Tensor) and true_boxes.ndim == 2 and true_boxes.size(0) > 0:
        # Expand to calculate pairwise IoU losses
        pred_boxes_exp = pred_boxes.unsqueeze(1)  # (num_pred_boxes, 1, 4)
        true_boxes_exp = true_boxes.unsqueeze(0)  # (1, num_true_boxes, 4)
        giou_loss_matrix = giou_loss(pred_boxes_exp, true_boxes_exp)  # (num_pred_boxes, num_true_boxes)

        # Minimize GIoU for the best matching ground truth for each predicted box
        giou_loss_per_pred = giou_loss_matrix.min(dim=1).values  # (num_pred_boxes,)
        giou_losses.append(giou_loss_per_pred.mean())  # Average over all predictions
    else:
        giou_losses.append(torch.tensor(0.0, device=pred_boxes.device))  # No predictions or targets
    # Classification Loss (Focal Loss)
    if pred_labels.shape[0] > 0 and isinstance(true_boxes, torch.Tensor) and true_boxes.ndim == 2 and true_boxes.size(0) > 0:
        classification_loss = focal_loss(pred_labels, true_labels, alpha, gamma)
        classification_losses.append(classification_loss)
    else:
        classification_losses.append(torch.tensor(0.0, device=pred_labels.device))  # No predictions or targets

    # Combine losses with weights
    total_giou_loss = torch.stack(giou_losses).mean()
    total_classification_loss = torch.stack(classification_losses).mean()
    total_loss = giou_weight * total_giou_loss + classification_weight * total_classification_loss
    
    return total_loss

def giou_loss(pred_boxes, true_boxes):
    """
    Generalized IoU loss for bounding boxes.

    Args:
        pred_boxes: Predicted bounding boxes (batch_size, num_boxes, 4).
        true_boxes: Ground truth bounding boxes (batch_size, num_boxes, 4).

    Returns:
        giou: Generalized IoU loss (batch_size, num_boxes).
    """
    # Calculate the coordinates of the smallest enclosing box
    pred_x1, pred_y1, pred_x2, pred_y2 = torch.split(pred_boxes, 1, dim=-1)
    true_x1, true_y1, true_x2, true_y2 = torch.split(true_boxes, 1, dim=-1)
    # Calculate the intersection area
    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    # Calculate the areas of the predicted and true boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    # Calculate the union area
    union_area = pred_area + true_area - inter_area
    # IoU: intersection / union
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    # Calculate the area of the smallest enclosing box (C)
    enclose_x1 = torch.min(pred_x1, true_x1)
    enclose_y1 = torch.min(pred_y1, true_y1)
    enclose_x2 = torch.max(pred_x2, true_x2)
    enclose_y2 = torch.max(pred_y2, true_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    # GIoU loss
    giou = iou - (enclose_area - union_area) / torch.clamp(enclose_area, min=1e-6)
    return 1 - giou  # GIoU loss is 1 - IoU-based value

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for multi-class classification.

    Args:
        predictions: Predicted class probabilities (batch_size, num_boxes, num_classes).
        targets: Ground truth labels (batch_size, num_boxes).
        alpha: Balancing factor for class imbalance.
        gamma: Focusing parameter.

    Returns:
        focal_loss: Computed focal loss.
    """
    predictions = predictions.float()
    # Apply softmax to get probabilities
    probs = torch.softmax(predictions, dim=-1)
    # Gather the predicted probabilities for the true class labels
    p_t = probs.gather(-1, targets.unsqueeze(-1))
    # Compute the focal loss
    loss = -alpha * (1 - p_t) ** gamma * p_t.log()
    # Return the mean focal loss
    return loss.mean()