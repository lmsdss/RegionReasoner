
import argparse
import torch
import json
import numpy as np
import os
from PIL import Image as PILImage
from tqdm import tqdm
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.models.qwen_vl import QwenVLModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vision_reasoner")
    parser.add_argument("--model_path", type=str, default="/RegionReasoner/test/pretrained_models/RegionReasoner-7B")
    parser.add_argument("--task_router_model_path", type=str, default="/RegionReasoner/test/pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True, default="/RegionReasoner/test/detection_eval_results/test_multi")
    parser.add_argument("--test_data_path", type=str, default="/RegionReasoner/test/testdata/refcocog_multi_turn.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=True,
        help="Whether to save visualization results",
    )  # default: True
    parser.add_argument(
        "--vis_output_path",
        type=str,
        default="/RegionReasoner/test/detection_eval_results/test_multi/visualizations",
        help="Path to save visualizations",
    )
    parser.add_argument(
        "--binarize_bbox_iou",
        action="store_true",
        default=True,
        help="Whether to binarize bbox IoU (set to 1.0 if IoU > 0.5, else 0.0)",
    )

    # for parallel evaluation
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    return parser.parse_args()

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    # Calculate the intersection area of two bboxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of the two bboxes
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

def save_visualization(image, pred_bbox, gt_bbox, pred_mask, gt_mask, question, turn_idx, image_id, intersection, union, bbox_iou, output_path):
    """
    Save a minimal visualization: a single image with the predicted mask and bbox overlay.
    
    Args:
        image: PIL image
        pred_bbox: Predicted bbox [x1, y1, x2, y2]
        gt_bbox: Ground-truth bbox [x1, y1, x2, y2] (unused)
        pred_mask: Predicted segmentation mask
        gt_mask: Ground-truth segmentation mask (unused)
        question: Question text (unused)
        turn_idx: Turn index (unused)
        image_id: Image ID (unused)
        intersection: IoU intersection (unused)
        union: IoU union (unused)
        bbox_iou: Bbox IoU (unused)
        output_path: Output path
    """
    
    # Create a single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Show original image
    ax.imshow(image)
    
    # Overlay predicted mask (red)
    if pred_mask is not None and pred_mask.size > 0:
        pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4))
        pred_colored[pred_mask] = [1, 0, 0, 0.5]  # red, 50% alpha
        ax.imshow(pred_colored)
    
    # Draw predicted bbox (green)
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    
    # Hide axes
    ax.axis('off')
    
    # Layout
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f"Visualization saved: {output_path}")

def main():
    args = parse_args()
    
    # Visualization output path
    if args.save_visualizations:
        if args.vis_output_path is None:
            args.vis_output_path = os.path.join(args.output_path, "visualizations")
        os.makedirs(args.vis_output_path, exist_ok=True)
        print(f"Visualizations will be saved to: {args.vis_output_path}")
    
    # Initialize model
    if args.model == "qwen":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "qwen2":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "qwen2.5":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "vision_reasoner":
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                    task_router_model_path=args.task_router_model_path, 
                                    segmentation_model_path=args.segmentation_model_path)
    
    # Load dataset
    with open(args.test_data_path, "r") as f:
        dataset = json.load(f)
    
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    dataset = dataset[start_idx:end_idx]
    
    all_outputs = []
    
    # Prepare batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = dataset[i:i + args.batch_size]
        
        batch_images = [PILImage.open(item["image_path"]).convert("RGB") for item in batch_data]
        all_bboxes_info = []  # Store cumulative visualizations for all turns in this batch

        # Process each conversational turn (fixed batch processing logic)
        for item_idx, item in enumerate(batch_data):
            conversational_turns = item["conversational_turns"]
            context_bbox = None
            # Store predicted results per turn for later context
            predicted_results = {}
            
            for turn_idx, turn in enumerate(conversational_turns):
                print(f"=== Turn {turn_idx} ===")
                question = turn["question"]
                # If the turn has a context, replace the <box> tag with the actual bbox
                if "context_from_turn" in turn:
                    context_turn_idx = turn["context_from_turn"]
                    # Use the previous turn's predicted bbox as context (instead of ground truth)
                    if context_turn_idx in predicted_results and predicted_results[context_turn_idx]["bboxes"]:
                        context_bbox = predicted_results[context_turn_idx]["bboxes"][0]  # previous prediction
                        print(f"Using predicted bbox from turn {context_turn_idx} as context: {context_bbox}")
                    else:
                        # Fallback: use ground-truth bbox if no prediction is available (ideally avoid)
                        context_bbox = conversational_turns[context_turn_idx]["bboxes"][0]
                        print(f"Warning: no prediction for turn {context_turn_idx}; fallback to GT bbox: {context_bbox}")
                    question = question.replace("<box>{context_bbox}</box>", str(context_bbox))
                
                print(f"Complete Question: {question}")
                print(f"Context Bbox: {context_bbox}")
                print(f"Image ID: {item['image_id']}")

                # Run inference per image to avoid batch mix-ups
                current_image = batch_images[item_idx]
                output = model.segment_objects_batch([current_image], [question])
                # Extract the bbox and mask from the output
                bboxes = output[0]["bboxes"]
                masks = output[0]["masks"]
                
                # Compute IoU for this turn
                try:
                    # Get predicted mask (ensure numpy)
                    if isinstance(masks, list):
                        pred_mask = np.array(masks[0]) if masks else np.array([])
                    else:
                        pred_mask = np.array(masks)
                    
                    # Get ground-truth mask
                    if "masks" in turn and turn["masks"]:
                        gt_mask = np.array(turn["masks"][0])  # first mask in list
                        # print(f"Debug - gt_mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
                        # print(f"Debug - pred_mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
                    else:
                        print(f"Warning: missing mask data (image_id={item['image_id']}, turn={turn_idx})")
                        gt_mask = np.zeros_like(pred_mask) if pred_mask.size > 0 else np.array([])
                    
                    # Compute mask IoU
                    if pred_mask.size > 0 and gt_mask.size > 0:
                        try:
                            # Ensure boolean type
                            pred_mask = pred_mask.astype(bool)
                            gt_mask = gt_mask.astype(bool)
                            
                            # Check shape match
                            if pred_mask.shape != gt_mask.shape:
                                print(f"Warning: mask shape mismatch - pred={pred_mask.shape}, gt={gt_mask.shape}")
                                
                                # Use true image size as reference
                                image_path = item["image_path"]
                                with PILImage.open(image_path) as img:
                                    img_width, img_height = img.size
                                    print(f"  True image size: {img_width}x{img_height}")
                                
                                # Resize both masks to the true image size
                                target_shape = (img_height, img_width)
                                print(f"  Resizing both masks to image size: {target_shape}")
                                
                                import cv2
                                # Resize predicted mask
                                if pred_mask.shape != target_shape:
                                    pred_mask = cv2.resize(
                                        pred_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                    print(f"   pred_mask resized to: {pred_mask.shape}")
                                
                                # Resize ground-truth mask
                                if gt_mask.shape != target_shape:
                                    gt_mask = cv2.resize(
                                        gt_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                    print(f"    gt_mask resized to: {gt_mask.shape}")
                            
                            intersection, union = compute_iou(pred_mask, gt_mask)
                        except Exception as e:
                            print(f"Warning: failed to compute mask IoU - error: {e}")
                            intersection, union = 0, 0
                        
                        print(f"IoU result: intersection={intersection}, union={union}")
                    else:
                        print("Error: empty mask data")
                        intersection, union = 0, 1  # avoid division by zero
                        
                except Exception as e:
                    print(f"Error while processing masks: {e}. Using defaults.")
                    intersection, union = 0, 1

                bbox_iou = 0.0
                try:
                    # Compare with turn-level ground-truth bbox if available
                    if "bboxes" in turn and turn["bboxes"] and bboxes:
                        gt_bbox = turn["bboxes"][0]  # ground-truth bbox
                        
                        if args.binarize_bbox_iou:
                            # Binarize: set to 1.0 if any IoU > 0.5, else 0.0
                            for pred_bbox in bboxes:
                                if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                    bbox_iou = 1.0
                                    break
                        else:
                            # Use the maximum IoU across predicted bboxes
                            max_iou = 0.0
                            for pred_bbox in bboxes:
                                current_iou = compute_bbox_iou(pred_bbox, gt_bbox)
                                max_iou = max(max_iou, current_iou)
                            bbox_iou = max_iou
                except Exception as e:
                    print(f"Error while processing bboxes: {e}. Setting bbox_iou=0.")
                    bbox_iou = 0.0

                # Save visualization (if enabled)
                if args.save_visualizations:
                    try:
                        # Reuse the already-loaded image
                        vis_image = current_image
                        
                        # Gather bbox/mask
                        pred_bbox = bboxes[0] if bboxes else None
                        gt_bbox = turn["bboxes"][0] if "bboxes" in turn and turn["bboxes"] else None
                        
                        # Filename (include idx to avoid multi-process conflicts)
                        ann_id = turn['ann_ids'][0] if 'ann_ids' in turn and turn['ann_ids'] else 'unknown'
                        vis_filename = f"gpu{args.idx}_image_{item['image_id']}_turn_{turn_idx}_ann_{ann_id}.png"
                        vis_path = os.path.join(args.vis_output_path, vis_filename)
                        
                        # Save visualization
                        save_visualization(
                            image=vis_image,
                            pred_bbox=pred_bbox,
                            gt_bbox=gt_bbox, 
                            pred_mask=pred_mask if 'pred_mask' in locals() else None,
                            gt_mask=gt_mask if 'gt_mask' in locals() else None,
                            question=question,
                            turn_idx=turn_idx,
                            image_id=item["image_id"],
                            intersection=intersection,
                            union=union,
                            bbox_iou=bbox_iou,
                            output_path=vis_path
                        )
                    except Exception as e:
                        print(f"Error while saving visualization: {e}")

                # Store predictions for later turns' context
                predicted_results[turn_idx] = {
                    "bboxes": bboxes,
                    "masks": masks,
                    "thinking": output[0]["thinking"]
                }

                # Store results for this turn (keep output format unchanged)
                result_entry = {
                    "image_id": str(item["image_id"]),  # ensure string
                    "ann_id": turn["ann_ids"][0] if "ann_ids" in turn and turn["ann_ids"] else f"{item['image_id']}_turn_{turn_idx}",
                    "think": output[0]["thinking"],
                    "intersection": int(intersection),
                    "union": int(union),
                    "bbox_iou": float(bbox_iou)
                }
                    
                all_outputs.append(result_entry)

    # Save results (moved outside the batch loop)
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()