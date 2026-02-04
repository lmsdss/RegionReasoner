import os
import json
import glob
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics_by_turns(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # Track the turn index for each image_id
    image_turn_counter = defaultdict(int)  # {image_id: current_turn_number}
    
    # Store items grouped by turn
    turns_data = defaultdict(list)  # {turn_number: [items]}
    
    # for calculating think text length by turns
    turns_think_lengths = defaultdict(list)  # {turn_number: [lengths]}
    
    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            image_id = item['image_id']
            
            # Determine which turn the current item belongs to
            current_turn = image_turn_counter[image_id] + 1
            image_turn_counter[image_id] = current_turn
            
            # Add the item to its corresponding turn bucket
            item_with_turn = item.copy()
            item_with_turn['turn'] = current_turn
            turns_data[current_turn].append(item_with_turn)
            
            # Calculate think text length if available
            if 'think' in item and item['think']:
                turns_think_lengths[current_turn].append(len(item['think']))
    
    # Print dataset overview
    print(f"\n=================== Dataset Overview ===================")
    total_images = len(image_turn_counter)
    max_turns = max(image_turn_counter.values()) if image_turn_counter else 0
    total_items = sum(len(items) for items in turns_data.values())
    
    print(f"Total unique images: {total_images}")
    print(f"Maximum turns per image: {max_turns}")
    print(f"Total evaluation items: {total_items}")
    
    # Count how many images appear in each turn
    turn_distribution = defaultdict(int)
    for image_id, max_turn in image_turn_counter.items():
        for turn in range(1, max_turn + 1):
            turn_distribution[turn] += 1
    
    print(f"\nTurn distribution:")
    for turn in sorted(turn_distribution.keys()):
        count = turn_distribution[turn]
        percentage = count / total_images * 100 if total_images > 0 else 0
        print(f"  Turn {turn}: {count} images ({percentage:.1f}%)")
    print(f"========================================================\n")
    
    # Compute metrics turn by turn
    for turn in sorted(turns_data.keys()):
        items = turns_data[turn]
        
        print(f"=================== Turn {turn} Results ===================")
        print(f"Number of items in Turn {turn}: {len(items)}")
        
        # Calculate think text metrics for this turn
        if turn in turns_think_lengths and turns_think_lengths[turn]:
            think_lengths = turns_think_lengths[turn]
            avg_think_length = sum(think_lengths) / len(think_lengths)
            min_think_length = min(think_lengths)
            max_think_length = max(think_lengths)
            # print(f"\nThink Text Statistics for Turn {turn}:")
            # print(f"  Number of think texts: {len(think_lengths)}")
            # print(f"  Average think text length: {avg_think_length:.2f} characters")
            # print(f"  Minimum think text length: {min_think_length} characters")
            # print(f"  Maximum think text length: {max_think_length} characters")
        
        # Compute metrics for this turn
        total_intersection = 0
        total_union = 0
        total_bbox_iou = 0
        all_ious = []
        cnt = 0
        
        for item in items:
            intersection = item['intersection']
            union = item['union']
            
            # calculate IoU of each item
            iou = intersection / union if union > 0 else 0
            all_ious.append({
                'image_id': item['image_id'],
                'iou': iou
            })
            
            # accumulate total intersection and union
            total_intersection += intersection
            total_union += union
            total_bbox_iou += item['bbox_iou']
            cnt += 1
        
        # calculate metrics for this turn
        gIoU = np.mean([item['iou'] for item in all_ious]) if all_ious else 0
        cIoU = total_intersection / total_union if total_union > 0 else 0
        bbox_iou = total_bbox_iou / cnt if cnt > 0 else 0
        
        # print the results for this turn
        print(f"\nMetrics for Turn {turn}:")
        print(f"  gIoU (average of per image IoU): {gIoU:.4f}")
        print(f"  cIoU (total_intersection / total_union): {cIoU:.4f}")
        print(f"  bbox_AP (average of per image bbox_AP): {bbox_iou:.4f}")
        
        # Additional statistics
        print(f"\nDetailed Statistics for Turn {turn}:")
        # print(f"  Total intersection: {total_intersection}")
        # print(f"  Total union: {total_union}")
        # print(f"  Average intersection per item: {total_intersection/cnt:.2f}")
        # print(f"  Average union per item: {total_union/cnt:.2f}")
        
        # IoU distribution stats
        if all_ious:
            ious_values = [item['iou'] for item in all_ious]
            # print(f"  IoU distribution:")
            # print(f"    Min IoU: {min(ious_values):.4f}")
            # print(f"    Max IoU: {max(ious_values):.4f}")
            # print(f"    Median IoU: {np.median(ious_values):.4f}")
            # print(f"    Std IoU: {np.std(ious_values):.4f}")
            
            # Pass rates under different IoU thresholds
            thresholds = [0.3, 0.5, 0.7, 0.9]
            print(f"  Pass rates at different thresholds:")
            for threshold in thresholds:
                pass_count = sum(1 for iou in ious_values if iou > threshold)
                pass_rate = pass_count / len(ious_values) * 100
                print(f"    IoU > {threshold}: {pass_rate:.1f}% ({pass_count}/{len(ious_values)})")
        
        print(f"======================================================\n")
    
    # Compute overall metrics (all turns combined)
    print(f"=================== Overall Results (All Turns) ===================")
    all_items = []
    for items in turns_data.values():
        all_items.extend(items)
    
    total_intersection = sum(item['intersection'] for item in all_items)
    total_union = sum(item['union'] for item in all_items)
    total_bbox_iou = sum(item['bbox_iou'] for item in all_items)
    
    all_ious = []
    for item in all_items:
        intersection = item['intersection']
        union = item['union']
        iou = intersection / union if union > 0 else 0
        all_ious.append({'image_id': item['image_id'], 'iou': iou})
    
    overall_gIoU = np.mean([item['iou'] for item in all_ious]) if all_ious else 0
    overall_cIoU = total_intersection / total_union if total_union > 0 else 0
    overall_bbox_iou = total_bbox_iou / len(all_items) if all_items else 0
    
    print(f"Overall gIoU (average of per image IoU): {overall_gIoU:.4f}")
    print(f"Overall cIoU (total_intersection / total_union): {overall_cIoU:.4f}")
    print(f"Overall bbox_AP (average of per image bbox_AP): {overall_bbox_iou:.4f}")
    print(f"================================================================\n")
    
    # Turn-by-turn comparison
    if len(turns_data) > 1:
        print(f"=================== Turn-by-Turn Comparison ===================")
        print(f"{'Turn':<6} {'gIoU':<8} {'cIoU':<8} {'bbox_AP':<8} {'Items':<6}")
        print("-" * 50)
        
        for turn in sorted(turns_data.keys()):
            items = turns_data[turn]
            
            # Recompute per-turn metrics (simplified)
            total_intersection = sum(item['intersection'] for item in items)
            total_union = sum(item['union'] for item in items)
            total_bbox_iou = sum(item['bbox_iou'] for item in items)
            
            turn_ious = []
            for item in items:
                intersection = item['intersection']
                union = item['union']
                iou = intersection / union if union > 0 else 0
                turn_ious.append(iou)
            
            turn_gIoU = np.mean(turn_ious) if turn_ious else 0
            turn_cIoU = total_intersection / total_union if total_union > 0 else 0
            turn_bbox_iou = total_bbox_iou / len(items) if items else 0
            
            print(f"{turn:<6} {turn_gIoU:<8.4f} {turn_cIoU:<8.4f} {turn_bbox_iou:<8.4f} {len(items):<6}")
        
        print("=" * 50)

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics_by_turns(args.output_dir)

    
