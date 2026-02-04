import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    total_intersection = 0
    total_union = 0
    all_ious = []
    
    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
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
    
    # calculate gIoU
    gIoU = np.mean([item['iou'] for item in all_ious])
    # calculate cIoU
    cIoU = total_intersection / total_union if total_union > 0 else 0
    
    # print the results
    print(f"gIoU (average of per image IoU): {gIoU:.4f}")
    print(f"cIoU (total_intersection / total_union): {cIoU:.4f}")
    

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)
