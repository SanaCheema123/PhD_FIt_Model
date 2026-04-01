import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_captioning(predictions, targets, idx2word):
    # Placeholder for captioning metrics
    # In a real implementation, would use BLEU, CIDEr, METEOR, etc.
    return {
        'bleu4': 0.0,
        'cider': 0.0,
        'meteor': 0.0
    }

def evaluate_classification(predictions, targets):
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    target_classes = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average='macro', zero_division=0)
    recall = recall_score(target_classes, pred_classes, average='macro', zero_division=0)
    f1 = f1_score(target_classes, pred_classes, average='macro', zero_division=0)
    
    # Calculate top-5 accuracy
    top5_acc = 0.0
    if predictions.shape[1] >= 5:
        top5_preds = torch.topk(predictions, k=5, dim=1)[1].cpu().numpy()
        top5_acc = np.mean([target in preds for target, preds in zip(target_classes, top5_preds)])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top5_accuracy': top5_acc
    }

def evaluate_detection(predictions, targets):
    # Placeholder for detection metrics
    # In a real implementation, would calculate mAP, IoU, etc.
    return {
        'map': 0.0,
        'iou': 0.0
    }

# utils/metrics.py (continuation from where it was cut off)

def evaluate_grounding(predictions, targets):
    # Calculate IoU between predicted and target boxes
    def calculate_iou(box1, box2):
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    
    # Calculate IoU for each sample in the batch
    ious = []
    for pred_box, target_box in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
        iou = calculate_iou(pred_box, target_box)
        ious.append(iou)
    
    # Calculate accuracy@0.5 (IoU > 0.5)
    accuracy = sum(1 for iou in ious if iou > 0.5) / len(ious) if ious else 0.0
    
    return {
        'iou': np.mean(ious) if ious else 0.0,
        'accuracy@0.5': accuracy
    }

def evaluate_segmentation(predictions, targets):
    # Calculate intersection over union for segmentation
    predictions = torch.sigmoid(predictions) > 0.5
    intersection = torch.logical_and(predictions, targets).sum().item()
    union = torch.logical_or(predictions, targets).sum().item()
    iou = intersection / union if union > 0 else 0.0
    
    # Calculate Dice coefficient
    dice = (2 * intersection) / (predictions.sum().item() + targets.sum().item()) if (predictions.sum().item() + targets.sum().item()) > 0 else 0.0
    
    return {
        'iou': iou,
        'dice': dice
    }
