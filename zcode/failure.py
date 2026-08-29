import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from sklearn.model_selection import train_test_split

# Global directories
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
        img = Image.open(img_path).convert("RGB")
        boxes = self.data.iloc[idx, 1:5].values.reshape(-1, 4).astype(float)
        label = self.data.iloc[idx, 5]
        labels = torch.tensor(
            [['can', 'bottle', 'container'].index(label)], dtype=torch.int64
        )
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": labels}
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class EvaluationDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        img = Image.open(img_path).convert("RGB")
        boxes = self.data.iloc[idx, 1:5].values.reshape(-1, 4).astype(float)
        labels = []
        labels.append(['can', 'bottle', 'container'].index(self.data.iloc[idx, 5]))
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": labels_tensor
        }
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def update_estimations(past, new):
    prev = past[0]
    for i in range(len(past)):
        (past[i], prev) = (prev, past[i])
    past[0] = new
    sum = 0
    for i in past:
        sum += i
    return sum / len(past)

def multi_task_loss(output, target, giou_weight=1.0, classification_weight=1.0):
    giou_losses = []
    classification_losses = []
    pred_boxes = output["boxes"]
    true_boxes = target["boxes"]
    pred_scores = output["scores"]  # Use scores instead of labels
    true_labels = target["labels"]

    # Localization - Generalized IoU Loss
    if pred_boxes.shape[0] > 0 and true_boxes.shape[0] > 0:
        pred_boxes_exp = pred_boxes.unsqueeze(1)
        true_boxes_exp = true_boxes.unsqueeze(0)
        giou_loss_matrix = giou_loss(pred_boxes_exp, true_boxes_exp)
        giou_loss_per_pred = giou_loss_matrix.min(dim=1).values
        giou_losses.append(giou_loss_per_pred.mean())
    else:
        giou_losses.append(torch.tensor(0.0, device=pred_boxes.device))

    # Classification - Focal Loss
    if pred_scores.shape[0] > 0:
        classification_losses.append(focal_loss(pred_scores, true_labels))
    else:
        classification_losses.append(torch.tensor(0.0, device=pred_scores.device))
    total_loss = giou_weight * torch.stack(giou_losses).mean() + classification_weight * torch.stack(classification_losses).mean()
    return total_loss

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    num_classes = predictions.shape[-1]
    if predictions.dtype != torch.float32:
        predictions = predictions.float()
    if targets.dim() == 1:  
        targets = targets.unsqueeze(1) 
    elif targets.dim() == 0: 
        targets = targets.unsqueeze(0)
    target_one_hot = torch.zeros(predictions.size(0), num_classes, device=predictions.device)
    target_one_hot.scatter_(1, targets, 1)
    probs = torch.softmax(predictions, dim=-1)
    p_t = (target_one_hot * probs).sum(dim=1)
    loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t + 1e-10)
    return loss.mean()

def giou_loss(pred_boxes, true_boxes):
    pred_x1, pred_y1, pred_x2, pred_y2 = torch.split(pred_boxes, 1, dim=-1)
    true_x1, true_y1, true_x2, true_y2 = torch.split(true_boxes, 1, dim=-1)
    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    union_area = pred_area + true_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    enclose_x1 = torch.min(pred_x1, true_x1)
    enclose_y1 = torch.min(pred_y1, true_y1)
    enclose_x2 = torch.max(pred_x2, true_x2)
    enclose_y2 = torch.max(pred_y2, true_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    giou = iou - (enclose_area - union_area) / torch.clamp(enclose_area, min=1e-6)
    return 1 - giou

def validate(model, validation_loader, device, multi_task_loss_fn):
    model.eval()
    val_loss = 0.0
    num_batches = len(validation_loader)
    with torch.no_grad():
        for idx, [images, targets] in enumerate(validation_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            batch_loss = sum(multi_task_loss_fn(output, target, 0.65, 0.35) for output, target in zip(outputs, targets))
            val_loss += batch_loss.item()
    avg_val_loss = val_loss / num_batches
    return avg_val_loss

def do_training(band, bs, test=None, path=None):
    train_data = pd.read_csv(os.path.join(DIRS[0], DIRS[2][0], f"train_{band}.csv"))
    test_path = os.path.join(DIRS[0], DIRS[2][0], f"test_{band}.csv")
    if test:
        train_data, _ = train_test_split(train_data, test_size=test[0])
    model_path = path if path else os.path.join(DIRS[0], DIRS[2][0], f"b{bs}_pth", f"model_{band}.pth")
    img_dir = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], f"chips_{band}")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ObjectDetectionDataset(train_data, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), pin_memory=True)
    vs = EvaluationDataset(test_path, img_dir, transform=transform)
    vd = DataLoader(vs, batch_size=bs, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_batches = len(dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssdlite320_mobilenet_v3_large(pretrained=True).to(device)
    num_classes = 3
    model.head.classification_head.num_classes = num_classes

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    num_epochs = test[1] if test else 250
    at = time.perf_counter()
    bt = time.perf_counter()
    pastEpochEstimations = [0,0,0]
    pastTrainingEstimations = [0,0,0,0,0,0,0,0,0,0]
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader):
            bt = time.perf_counter()
            batchTime = bt - at
            epochTimeLeft = update_estimations(pastEpochEstimations, batchTime * (num_batches - batch_idx))
            trainingTimeLeft = update_estimations(pastTrainingEstimations, epochTimeLeft + (num_epochs - epoch) * batchTime * num_batches)
            print(f"                                                                                         Training Model_{bs}_{band}: {{Batch {(batch_idx+1):03}/{num_batches} - {batchTime:09.8f}s | Epoch {epoch:02}/{num_epochs} ETA - 00:{int(epochTimeLeft // 60):02}:{int(epochTimeLeft - epochTimeLeft // 60 * 60):02} | Training ETA - {int(trainingTimeLeft // 3600):02}:{int((trainingTimeLeft - trainingTimeLeft // 3600 * 3600) // 60):02}:{int(trainingTimeLeft - trainingTimeLeft // 3600 * 3600 - (trainingTimeLeft - trainingTimeLeft // 3600 * 3600) // 60 * 60):02} | {100 * (batch_idx + num_batches * (epoch - 1)) / (num_epochs * num_batches):08.4f}%}} ", end="\r")
            at = time.perf_counter()

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        train_loss = running_loss / num_batches
        print(f"Training Loss: {train_loss:.4f}", end = ' | ')

        # Validation
        if epoch % 3 == 0:
            val_loss = validate(model, vd, device, multi_task_loss)
            print(f"Validation Loss: {val_loss:.4f}", end='')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print(". Model saved", end='')
            else:
                patience_counter += 1
                print(f". No val improvement: {patience_counter}/{patience}", end='')

            if patience_counter >= patience:
                print(f"Early stopping at Epoch {epoch} due to no improvement in Validation Loss.")
                break
        print('')
        scheduler.step(train_loss)

    print(f'--> model_{band}.pth successfully saved to b{bs}_pth.')