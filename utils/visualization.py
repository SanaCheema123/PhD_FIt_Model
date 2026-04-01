import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.utils import make_grid

def visualize_predictions(images, predictions, targets=None, task='all'):
    """
    Visualize model predictions for different tasks
    
    Args:
        images: Batch of images [B, C, H, W]
        predictions: Dict of predictions for different tasks
        targets: Dict of targets for different tasks (optional)
        task: Specific task to visualize or 'all'
    """
    # Convert images from tensor to numpy and denormalize
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
        # Denormalize
        images = images * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        images = torch.clamp(images, 0, 1)
        
        # Convert to numpy
        images = images.permute(0, 2, 3, 1).numpy()
    
    batch_size = len(images)
    
    if task == 'all':
        fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
        
        # If batch size is 1, wrap axes in a list
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Captioning
            if 'captioning' in predictions:
                # Placeholder for caption visualization
                axes[i, 1].imshow(images[i])
                axes[i, 1].set_title('Captioning')
                axes[i, 1].axis('off')
            else:
                axes[i, 1].axis('off')
            
            # Detection/Grounding
            if 'detection' in predictions or 'grounding' in predictions:
                axes[i, 2].imshow(images[i])
                
                # Draw bounding boxes
                if 'detection' in predictions:
                    boxes = predictions['detection'][0][i].detach().cpu().numpy()  # class_preds, box_preds
                    for box in boxes:
                        x, y, w, h = box
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        axes[i, 2].add_patch(rect)
                
                if 'grounding' in predictions:
                    box = predictions['grounding'][i].detach().cpu().numpy()
                    x, y, w, h = box
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    axes[i, 2].add_patch(rect)
                
                axes[i, 2].set_title('Detection/Grounding')
                axes[i, 2].axis('off')
            else:
                axes[i, 2].axis('off')
            
            # Segmentation
            if 'segmentation' in predictions:
                mask = predictions['segmentation'][0][i, 0].detach().cpu().numpy()  # mask_pred, class_pred
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
                
                # Create overlay
                alpha = 0.5
                overlay = images[i] * (1 - alpha) + mask * alpha
                
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title('Segmentation')
                axes[i, 3].axis('off')
            else:
                axes[i, 3].axis('off')
            
            # Classification
            if 'classification' in predictions:
                axes[i, 4].imshow(images[i])
                
                # Get top-5 predictions
                if isinstance(predictions['classification'], torch.Tensor):
                    probs = F.softmax(predictions['classification'][i], dim=0)
                    top5_prob, top5_idx = torch.topk(probs, 5)
                    
                    # Display class probabilities
                    class_text = '\n'.join([f'Class {idx.item()}: {prob.item():.3f}' for idx, prob in zip(top5_idx, top5_prob)])
                    axes[i, 4].text(0, 0, class_text, transform=axes[i, 4].transAxes)
                
                axes[i, 4].set_title('Classification')
                axes[i, 4].axis('off')
            else:
                axes[i, 4].axis('off')
    
    else:
        # Visualize specific task
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, 4 * batch_size))
        
        # If batch size is 1, wrap axes in a list
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Task-specific visualization
            if task == 'captioning':
                # Placeholder for caption visualization
                axes[i, 1].imshow(images[i])
                axes[i, 1].set_title('Captioning')
                axes[i, 1].axis('off')
            
            elif task in ['detection', 'grounding']:
                axes[i, 1].imshow(images[i])
                
                # Draw bounding boxes
                if task == 'detection' and 'detection' in predictions:
                    boxes = predictions['detection'][0][i].detach().cpu().numpy()
                    for box in boxes:
                        x, y, w, h = box
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        axes[i, 1].add_patch(rect)
                
                if task == 'grounding' and 'grounding' in predictions:
                    box = predictions['grounding'][i].detach().cpu().numpy()
                    x, y, w, h = box
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    axes[i, 1].add_patch(rect)
                
                axes[i, 1].set_title(task.capitalize())
                axes[i, 1].axis('off')
            
            elif task == 'segmentation':
                if 'segmentation' in predictions:
                    mask = predictions['segmentation'][0][i, 0].detach().cpu().numpy()
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    
                    # Create overlay
                    alpha = 0.5
                    overlay = images[i] * (1 - alpha) + mask * alpha
                    
                    axes[i, 1].imshow(overlay)
                    axes[i, 1].set_title('Segmentation')
                    axes[i, 1].axis('off')
            
            elif task == 'classification':
                if 'classification' in predictions:
                    axes[i, 1].imshow(images[i])
                    
                    # Get top-5 predictions
                    if isinstance(predictions['classification'], torch.Tensor):
                        probs = F.softmax(predictions['classification'][i], dim=0)
                        top5_prob, top5_idx = torch.topk(probs, 5)
                        
                        # Display class probabilities
                        class_text = '\n'.join([f'Class {idx.item()}: {prob.item():.3f}' for idx, prob in zip(top5_idx, top5_prob)])
                        axes[i, 1].text(0, 0, class_text, transform=axes[i, 1].transAxes)
                    
                    axes[i, 1].set_title('Classification')
                    axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig
