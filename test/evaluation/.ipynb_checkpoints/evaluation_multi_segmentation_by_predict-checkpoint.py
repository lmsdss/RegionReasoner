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

# 将父目录添加到Python路径中，以便导入模型模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.models.qwen_vl import QwenVLModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vision_reasoner")
    parser.add_argument("--model_path", type=str, default="/sunwenfang/VisionReasoner/pretrained_models/VisionReasoner-7B")
    parser.add_argument("--task_router_model_path", type=str, default="/sunwenfang/VisionReasoner/pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True, default="/sunwenfang/VisionReasoner/detection_eval_results/test_multi")
    parser.add_argument("--test_data_path", type=str, default="/sunwenfang/VisionReasoner/evaluation/test1.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_visualizations", action="store_true", default=True, help="是否保存可视化结果") # 默认为True False
    parser.add_argument("--vis_output_path", type=str, default="/sunwenfang/VisionReasoner/detection_eval_results/test_multi/0722visualizations", help="可视化结果保存路径")
    parser.add_argument("--binarize_bbox_iou", action="store_true", default=True, help="是否对bbox_iou进行二值化处理(IoU>0.5设为1.0)")

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
    保存简洁的可视化结果，只显示一个图片包含预测的mask和bbox
    
    Args:
        image: PIL图像
        pred_bbox: 预测的边界框 [x1, y1, x2, y2]
        gt_bbox: 真实的边界框 [x1, y1, x2, y2] (未使用)
        pred_mask: 预测的分割掩码
        gt_mask: 真实的分割掩码 (未使用)
        question: 问题文本 (未使用)
        turn_idx: 轮次索引 (未使用)
        image_id: 图像ID (未使用)
        intersection: IoU交集 (未使用)
        union: IoU并集 (未使用)
        bbox_iou: 边界框IoU (未使用)
        output_path: 保存路径
    """
    
    # 创建单个图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 显示原图
    ax.imshow(image)
    
    # 添加预测的分割掩码 - 红色
    if pred_mask is not None and pred_mask.size > 0:
        pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4))
        pred_colored[pred_mask] = [1, 0, 0, 0.5]  # 红色，50%透明度
        ax.imshow(pred_colored)
    
    # 添加预测的边界框 - 绿色
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    
    # 关闭坐标轴
    ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"可视化已保存: {output_path}")

def main():
    args = parse_args()
    
    # 设置可视化输出路径
    if args.save_visualizations:
        if args.vis_output_path is None:
            args.vis_output_path = os.path.join(args.output_path, "visualizations")
        os.makedirs(args.vis_output_path, exist_ok=True)
        print(f"可视化结果将保存到: {args.vis_output_path}")
    
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
        all_bboxes_info = []  # Store the cumulative visualizations for all turns in this batch

        # Process each conversational turn - 修复批量处理逻辑
        for item_idx, item in enumerate(batch_data):
            conversational_turns = item["conversational_turns"]
            context_bbox = None
            # 存储每轮的预测结果，用于后续轮次的上下文
            predicted_results = {}
            
            for turn_idx, turn in enumerate(conversational_turns):
                question = turn["question"]
                # If the turn has a context, replace the <box> tag with the actual bbox
                if "context_from_turn" in turn:
                    context_turn_idx = turn["context_from_turn"]
                    # 使用前一轮的实际预测结果作为上下文，而不是ground truth
                    if context_turn_idx in predicted_results and predicted_results[context_turn_idx]["bboxes"]:
                        context_bbox = predicted_results[context_turn_idx]["bboxes"][0]  # 使用前一轮的预测结果
                        print(f"使用第{context_turn_idx}轮的预测bbox作为上下文: {context_bbox}")
                    else:
                        # 如果前一轮没有预测结果，回退到使用ground truth（但这种情况应该避免）
                        context_bbox = conversational_turns[context_turn_idx]["bboxes"][0]
                        print(f"警告：第{context_turn_idx}轮没有预测结果，回退使用ground truth bbox: {context_bbox}")
                    question = question.replace("<box>{context_bbox}</box>", str(context_bbox))
                
                print(f"=== Turn {turn_idx} ===")
                print(f"Complete Question: {question}")
                print(f"Context Bbox: {context_bbox}")
                print(f"Image ID: {item['image_id']}")

                # 为当前图像单独进行推理，避免批量处理混乱
                current_image = batch_images[item_idx]
                output = model.segment_objects_batch([current_image], [question])
                # Extract the bbox and mask from the output
                bboxes = output[0]["bboxes"]
                masks = output[0]["masks"]
                
                # 调试信息：检查masks格式
                # print(f"Debug - masks type: {type(masks)}, shape: {getattr(masks, 'shape', 'no shape')}")
                # if hasattr(masks, 'shape'):
                    # print(f"Debug - masks dtype: {masks.dtype}, min: {masks.min()}, max: {masks.max()}")
                # if isinstance(masks, list) and masks:
                    # print(f"Debug - first mask type: {type(masks[0])}, shape: {getattr(masks[0], 'shape', 'no shape')}")

                # Compute IoU for this turn
                try:
                    # 获取预测的mask - 确保是numpy数组格式
                    if isinstance(masks, list):
                        pred_mask = np.array(masks[0]) if masks else np.array([])
                    else:
                        pred_mask = np.array(masks)
                    
                    # 获取真实的mask
                    if "masks" in turn and turn["masks"]:
                        gt_mask = np.array(turn["masks"][0])  # 使用masks列表中的第一个mask
                        # print(f"Debug - gt_mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
                        # print(f"Debug - pred_mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
                    else:
                        print(f"警告: image_id {item['image_id']} turn {turn_idx} 缺少mask数据")
                        gt_mask = np.zeros_like(pred_mask) if pred_mask.size > 0 else np.array([])
                    
                    # 计算mask IoU
                    if pred_mask.size > 0 and gt_mask.size > 0:
                        try:
                            # 确保是布尔类型
                            pred_mask = pred_mask.astype(bool)
                            gt_mask = gt_mask.astype(bool)
                            
                            # 检查mask形状是否匹配
                            if pred_mask.shape != gt_mask.shape:
                                print(f"警告: mask形状不匹配 - pred: {pred_mask.shape}, gt: {gt_mask.shape}")
                                
                                # 获取图像真实尺寸作为参考
                                image_path = item["image_path"]
                                with PILImage.open(image_path) as img:
                                    img_width, img_height = img.size
                                    print(f"  图像真实尺寸: {img_width}x{img_height}")
                                
                                # 优先将两个mask都调整到图像的真实尺寸
                                target_shape = (img_height, img_width)
                                print(f"  将两个mask统一调整到图像尺寸: {target_shape}")
                                
                                import cv2
                                # 调整预测mask
                                if pred_mask.shape != target_shape:
                                    pred_mask = cv2.resize(
                                        pred_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                    print(f"   pred_mask调整为: {pred_mask.shape}")
                                
                                # 调整真实mask 
                                if gt_mask.shape != target_shape:
                                    gt_mask = cv2.resize(
                                        gt_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                    print(f"    gt_mask调整为: {gt_mask.shape}")
                            
                            intersection, union = compute_iou(pred_mask, gt_mask)
                        except Exception as e:
                            print(f"警告: mask计算IoU失败 - 错误: {e}")
                            intersection, union = 0, 0
                        
                        print(f" IoU计算结果: intersection={intersection}, union={union}")
                    else:
                        print("错误: mask数据为空")
                        intersection, union = 0, 1  # 避免除零错误
                        
                except Exception as e:
                    print(f"处理mask时出错: {e}, 使用默认值")
                    intersection, union = 0, 1

                bbox_iou = 0.0
                try:
                    # 尝试获取当前轮次的bbox数据进行比较
                    if "bboxes" in turn and turn["bboxes"] and bboxes:
                        gt_bbox = turn["bboxes"][0]  # 使用ground truth bbox
                        
                        if args.binarize_bbox_iou:
                            # 二值化逻辑：IoU > 0.5 设为1.0，否则为0.0
                            for pred_bbox in bboxes:
                                if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                    bbox_iou = 1.0
                                    break
                        else:
                            # 计算真实IoU值：取所有预测框中的最大IoU
                            max_iou = 0.0
                            for pred_bbox in bboxes:
                                current_iou = compute_bbox_iou(pred_bbox, gt_bbox)
                                max_iou = max(max_iou, current_iou)
                            bbox_iou = max_iou
                except Exception as e:
                    print(f"处理bbox时出错: {e}, bbox_iou设为0")
                    bbox_iou = 0.0

                # 保存可视化结果（如果启用）
                if args.save_visualizations:
                    try:
                        # 使用已加载的当前图像，避免重复加载
                        vis_image = current_image
                        
                        # 获取边界框和掩码数据
                        pred_bbox = bboxes[0] if bboxes else None
                        gt_bbox = turn["bboxes"][0] if "bboxes" in turn and turn["bboxes"] else None
                        
                        # 构造文件名 - 添加进程ID避免多GPU冲突
                        ann_id = turn['ann_ids'][0] if 'ann_ids' in turn and turn['ann_ids'] else 'unknown'
                        vis_filename = f"gpu{args.idx}_image_{item['image_id']}_turn_{turn_idx}_ann_{ann_id}.png"
                        vis_path = os.path.join(args.vis_output_path, vis_filename)
                        
                        # 保存可视化
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
                        print(f"保存可视化时出错: {e}")

                # 存储当前轮的预测结果，用于后续轮次的上下文
                predicted_results[turn_idx] = {
                    "bboxes": bboxes,
                    "masks": masks,
                    "thinking": output[0]["thinking"]
                }

                # Store the results for this turn - 保持与原格式一致
                result_entry = {
                    "image_id": str(item["image_id"]),  # 确保是字符串格式
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