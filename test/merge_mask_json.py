import json
import sys
import numpy as np
import argparse

# Merge two files: add full ann_ids and masks into multi-turn conversation data.

def compute_bbox_iou(bbox1, bbox2):
    """Compute IoU (intersection-over-union) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def find_matching_annotation(target_bbox, merged_item, turn_idx, image_id):
    """
    Smart annotation matching: find the best ann_id and mask based on bbox similarity.
    
    Args:
        target_bbox: Target bbox [x1, y1, x2, y2]
        merged_item: Full merged item
        turn_idx: Current turn index
        image_id: Image ID
    
    Returns:
        tuple: (matched_ann_id, matched_mask)
    """
    if not target_bbox or not merged_item.get("bboxes"):
        return None, None
    
    available_bboxes = merged_item.get("bboxes", [])
    available_ann_ids = merged_item.get("ann_ids", [])
    available_masks = merged_item.get("masks", [])
    
    if not available_bboxes or not available_ann_ids or not available_masks:
        return None, None
    
    # Ensure all lists have consistent lengths
    min_length = min(len(available_bboxes), len(available_ann_ids), len(available_masks))
    if min_length == 0:
        return None, None
    
    # Compute IoU with all available bboxes
    best_iou = 0
    best_match_idx = -1
    iou_scores = []
    
    print(f"  Target bbox: {target_bbox}")
    for i in range(min_length):
        available_bbox = available_bboxes[i]
        iou = compute_bbox_iou(target_bbox, available_bbox)
        iou_scores.append(iou)
        print(f"  Available[{i}] bbox: {available_bbox}, IoU: {iou:.3f}, ann_id: {available_ann_ids[i]}")
        
        if iou > best_iou:
            best_iou = iou
            best_match_idx = i
    
    print(f"  Best match: idx={best_match_idx}, IoU={best_iou:.3f}")
    
    # Prefer the best IoU match first (even if IoU is not high)
    if best_match_idx >= 0:
        if best_iou > 0.3:  # Lower threshold for easier matching
            print(
                f"image_id {image_id} turn {turn_idx}: IoU match (IoU={best_iou:.3f}) -> ann_id {available_ann_ids[best_match_idx]}"
            )
            return available_ann_ids[best_match_idx], available_masks[best_match_idx]
        else:
            # Even if IoU is low, keep the best match as a fallback
            best_similarity_match = (available_ann_ids[best_match_idx], available_masks[best_match_idx])
    
    # If turn index is in range, check position-based match IoU
    if turn_idx < min_length:
        position_bbox = available_bboxes[turn_idx]
        position_iou = compute_bbox_iou(target_bbox, position_bbox)
        
        # Use position-based match if it has some IoU, or if all matches are very poor
        if position_iou > 0.1 or best_iou < 0.1:
            print(
                f"image_id {image_id} turn {turn_idx}: position match (IoU={position_iou:.3f}) -> ann_id {available_ann_ids[turn_idx]}"
            )
            return available_ann_ids[turn_idx], available_masks[turn_idx]
    
    # Use the best similarity match
    if best_match_idx >= 0:
        print(
            f"image_id {image_id} turn {turn_idx}: best similarity match (IoU={best_iou:.3f}) -> ann_id {available_ann_ids[best_match_idx]}"
        )
        return available_ann_ids[best_match_idx], available_masks[best_match_idx]
    
    # Final fallback: use the first available entry
    else:
        print(f"image_id {image_id} turn {turn_idx}: fallback -> ann_id {available_ann_ids[0]}")
        return available_ann_ids[0], available_masks[0]

def merge_datasets(test_json_path, merged_json_path, output_path):
    """
    Merge multi-turn conversation data with the full merged dataset.
    
    Args:
        test_json_path: Multi-turn conversation JSON path
        merged_json_path: Full merged JSON path
        output_path: Output path
    """
    
    print("Loading data files...")
    
    # Load multi-turn conversation data
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} multi-turn conversation samples")
    
    # Load full merged data
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"Loaded {len(merged_data)} full merged samples")
    
    # Build image_id -> merged item mapping
    merged_data_dict = {item['image_id']: item for item in merged_data}
    print("Built image_id mapping dictionary")
    
    # Merge
    merged_output = []
    missing_images = []
    
    for test_item in test_data:
        image_id = test_item['image_id']
        
        if image_id not in merged_data_dict:
            print(f"Warning: image_id {image_id} not found in full merged data")
            missing_images.append(image_id)
            continue
        
        merged_item = merged_data_dict[image_id]
        
        # Create output structure
        new_item = {
            "image_id": image_id,
            "conversational_turns": []
        }
        
        # If test_item has image_path, keep it
        if "image_path" in test_item:
            new_item["image_path"] = test_item["image_path"]
        else:
            # Otherwise, try to take it from merged_item, or generate a default path
            if "image_path" in merged_item:
                new_item["image_path"] = merged_item["image_path"]
            else:
                # Generate default image_path
                new_item["image_path"] = f"images/{image_id}.jpg"
        
        # Process each conversation turn
        for turn_idx, turn in enumerate(test_item["conversational_turns"]):
            new_turn = {
                "question": turn["question"],
                "bboxes": turn["bboxes"]
            }
            
            # Add context (if present)
            if "context_from_turn" in turn:
                new_turn["context_from_turn"] = turn["context_from_turn"]
            
            # Match ann_id and mask
            matched_ann_id, matched_mask = find_matching_annotation(
                turn["bboxes"][0] if turn["bboxes"] else None,
                merged_item,
                turn_idx,
                image_id
            )
            
            if matched_ann_id and matched_mask is not None:
                new_turn["ann_ids"] = [matched_ann_id]
                new_turn["masks"] = [matched_mask]
            else:
                print(f"Error: image_id {image_id} turn {turn_idx} cannot find a matching annotation")
                continue
            
            new_item["conversational_turns"].append(new_turn)
        
        merged_output.append(new_item)
    
    # Save merged output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_output, f, indent=2, ensure_ascii=False)
    
    print("\nMerge completed!")
    print(f"Output file: {output_path}")
    print(f"Successfully merged: {len(merged_output)} samples")
    if missing_images:
        print(f"Missing images: {len(missing_images)} - {missing_images}")
    
    return merged_output

def verify_merged_data(merged_data_path):
    """
    Verify the structure of the merged output data.
    """
    print("\nVerifying merged data...")
    
    with open(merged_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Check the structure of the first sample
    if data:
        first_item = data[0]
        print("\nFirst sample structure:")
        print(f"- image_id: {first_item['image_id']}")
        print(f"- image_path: {first_item['image_path']}")
        print(f"- conversational_turns count: {len(first_item['conversational_turns'])}")
        
        for i, turn in enumerate(first_item['conversational_turns']):
            print(f"  Turn {i}:")
            print(f"    - question: {turn['question']}")
            print(f"    - bboxes count: {len(turn['bboxes'])}")
            print(f"    - ann_ids: {turn.get('ann_ids', 'Missing!')}")
            print(f"    - masks present: {'masks' in turn}")
            if 'context_from_turn' in turn:
                print(f"    - context_from_turn: {turn['context_from_turn']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multi-turn questions JSON with mask/label JSON to produce a final *_multi_turn.json.",
    )
    parser.add_argument(
        "--question_json",
        default="test/testdata/refcocoplus_multi_question.json",
        help="Path to *_multi_question.json (questions).",
    )
    parser.add_argument(
        "--mask_json",
        default="test/testdata/refcocoplus_multi_mask.json",
        help="Path to *_multi_mask.json (labels: bboxes/ann_ids/masks).",
    )
    parser.add_argument(
        "--output_json",
        default="test/testdata/refcocoplus_multi_turn.json",
        help="Output path for *_multi_turn.json.",
    )
    args = parser.parse_args()

    test_json_path = args.question_json
    merged_json_path = args.mask_json
    output_path = args.output_json
    
    try:
        merged_data = merge_datasets(test_json_path, merged_json_path, output_path)
        
        verify_merged_data(output_path)
    except Exception as e:
        import traceback
        traceback.print_exc() 
