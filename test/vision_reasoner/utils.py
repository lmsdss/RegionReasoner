import matplotlib.pyplot as plt
import numpy as np

def visualize_results_enhanced(image, result, task_type, output_path):
    """
    Enhanced visualization with three-panel layout
    """
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # First panel: Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Second panel: Image with bounding boxes
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    if 'bboxes' in result and result['bboxes']:
        for bbox in result['bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    
    if 'points' in result and result['points']:
        for point in result['points']:
            plt.plot(point[0], point[1], 'go', markersize=8)  # Green point
    
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    
    # Third panel: Mask overlay (for segmentation tasks)
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)
    
    if task_type == 'segmentation' and 'masks' in result and result['masks'] is not None:
        mask = result['masks']
        if np.any(mask):
            mask_overlay = np.zeros_like(np.array(image))
            mask_overlay[mask] = [255, 0, 0]  # Red color for mask
            plt.imshow(mask_overlay, alpha=0.4)
    
    if task_type == 'detection' or task_type == 'counting':
        # For non-segmentation tasks, just show bounding boxes again
        if 'bboxes' in result and result['bboxes']:
            for bbox in result['bboxes']:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=True, edgecolor='red', facecolor='red', alpha=0.3)
                plt.gca().add_patch(rect)
    
    task_title = {
        'detection': 'Detection Overlay',
        'segmentation': 'Segmentation Mask',
        'counting': 'Counting Results',
        'qa': 'Visual QA'
    }
    
    plt.title(task_title.get(task_type, 'Results Overlay'))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()