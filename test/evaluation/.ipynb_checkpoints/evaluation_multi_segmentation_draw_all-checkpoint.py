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
    parser.add_argument("--save_visualizations", action="store_true", default=True, help="是否保存可视化结果")
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
    保存预测结果和真实标注的可视化对比图
    
    Args:
        image: PIL图像
        pred_bbox: 预测的边界框 [x1, y1, x2, y2]
        gt_bbox: 真实的边界框 [x1, y1, x2, y2]
        pred_mask: 预测的分割掩码
        gt_mask: 真实的分割掩码
        question: 问题文本
        turn_idx: 轮次索引
        image_id: 图像ID
        intersection: IoU交集
        union: IoU并集
        bbox_iou: 边界框IoU
        output_path: 保存路径
    """
    
    # 创建子图布局：原图 + 预测结果 + 真实标注 + 对比图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 处理长问题文本 - 自动换行
    max_line_length = 80
    if len(question) > max_line_length:
        # 将长问题分成多行
        words = question.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_line_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        question_display = "\n".join(lines)
    else:
        question_display = question
    
    fig.suptitle(f"Image {image_id} - Turn {turn_idx}\nQuestion: {question_display}", fontsize=12, y=0.98)
    
    # 1. 原图 + 预测边界框
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Predicted (BBox IoU: {bbox_iou:.3f})")
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none', label='Predicted')
        axes[0, 0].add_patch(rect)
    axes[0, 0].axis('off')
    axes[0, 0].legend()
    
    # 2. 原图 + 真实边界框
    axes[0, 1].imshow(image)
    axes[0, 1].set_title("Ground Truth")
    if gt_bbox is not None:
        x1, y1, x2, y2 = gt_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth')
        axes[0, 1].add_patch(rect)
    axes[0, 1].axis('off')
    axes[0, 1].legend()
    
    # 3. 分割掩码对比
    axes[1, 0].imshow(image)
    axes[1, 0].set_title(f"Segmentation Masks (IoU: {intersection/max(union, 1):.3f})")
    
    # 预测掩码 - 红色
    if pred_mask is not None and pred_mask.size > 0:
        pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4))
        pred_colored[pred_mask] = [1, 0, 0, 0.5]  # 红色，50%透明度
        axes[1, 0].imshow(pred_colored)
    
    # 真实掩码 - 绿色
    if gt_mask is not None and gt_mask.size > 0:
        gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4))
        gt_colored[gt_mask] = [0, 1, 0, 0.3]  # 绿色，30%透明度
        axes[1, 0].imshow(gt_colored)
    
    axes[1, 0].axis('off')
    
    # 添加图例
    red_patch = patches.Patch(color='red', alpha=0.5, label='Predicted Mask')
    green_patch = patches.Patch(color='green', alpha=0.3, label='Ground Truth Mask')
    axes[1, 0].legend(handles=[red_patch, green_patch], loc='upper right')
    
    # 4. 统计信息
    y_pos = 0.9
    axes[1, 1].text(0.05, y_pos, f"Metrics:", fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    y_pos -= 0.08
    axes[1, 1].text(0.05, y_pos, f"• Intersection: {intersection}", fontsize=11, transform=axes[1, 1].transAxes)
    y_pos -= 0.06
    axes[1, 1].text(0.05, y_pos, f"• Union: {union}", fontsize=11, transform=axes[1, 1].transAxes)
    y_pos -= 0.06
    axes[1, 1].text(0.05, y_pos, f"• Mask IoU: {intersection/max(union, 1):.4f}", fontsize=11, transform=axes[1, 1].transAxes)
    y_pos -= 0.06
    axes[1, 1].text(0.05, y_pos, f"• BBox IoU: {bbox_iou:.4f}", fontsize=11, transform=axes[1, 1].transAxes)
    y_pos -= 0.08
    
    if pred_bbox is not None:
        axes[1, 1].text(0.05, y_pos, f"• Pred BBox:", fontsize=10, fontweight='bold', transform=axes[1, 1].transAxes)
        y_pos -= 0.05
        axes[1, 1].text(0.05, y_pos, f"  [{pred_bbox[0]}, {pred_bbox[1]}, {pred_bbox[2]}, {pred_bbox[3]}]", 
                       fontsize=9, transform=axes[1, 1].transAxes)
        y_pos -= 0.06
    if gt_bbox is not None:
        axes[1, 1].text(0.05, y_pos, f"• GT BBox:", fontsize=10, fontweight='bold', transform=axes[1, 1].transAxes)
        y_pos -= 0.05
        axes[1, 1].text(0.05, y_pos, f"  [{gt_bbox[0]}, {gt_bbox[1]}, {gt_bbox[2]}, {gt_bbox[3]}]", 
                       fontsize=9, transform=axes[1, 1].transAxes)
        y_pos -= 0.08
    
    # 添加完整的问题文本
    if y_pos > 0.1:  # 确保有足够空间
        axes[1, 1].text(0.05, y_pos, f"Question:", fontsize=10, fontweight='bold', transform=axes[1, 1].transAxes)
        y_pos -= 0.05
        # 将问题文本分行显示
        question_words = question.split()
        current_line = ""
        for word in question_words:
            if len(current_line + " " + word) <= 30:  # 每行约30个字符
                current_line += (" " + word) if current_line else word
            else:
                if current_line and y_pos > 0.05:
                    axes[1, 1].text(0.05, y_pos, current_line, fontsize=8, transform=axes[1, 1].transAxes)
                    y_pos -= 0.04
                current_line = word
        if current_line and y_pos > 0.05:
            axes[1, 1].text(0.05, y_pos, current_line, fontsize=8, transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # 调整布局，为标题留出更多空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 为标题留出更多空间
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
#   print(f"可视化已保存: {output_path}")

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
        
        # Process each conversational turn
        for item in batch_data:
            conversational_turns = item["conversational_turns"]
            context_bbox = None
            
            for turn_idx, turn in enumerate(conversational_turns):
                question = turn["question"]
                # If the turn has a context, replace the <box> tag with the actual bbox
                if "context_from_turn" in turn:
                    context_turn_idx = turn["context_from_turn"]
                    context_bbox = conversational_turns[context_turn_idx]["bboxes"][0]  # Refer to the previous turn's bbox
                    question = question.replace("<box>{context_bbox}</box>", str(context_bbox))
                
                print(f"=== Turn {turn_idx} ===")
                print(f"Complete Question: {question}")
                print(f"Image ID: {item['image_id']}")
                print(f"Image Path: {item['image_path']}")

                # 修复：为当前item单独加载图片，确保图片和问题一一对应
                current_image = PILImage.open(item["image_path"]).convert("RGB")
#                 print(f"Loaded image size: {current_image.size}")
                
                # Perform inference for the current turn
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
                                # print(f"警告: mask形状不匹配 - pred: {pred_mask.shape}, gt: {gt_mask.shape}")
                                
                                # 获取图像真实尺寸作为参考
                                image_path = item["image_path"]
                                with PILImage.open(image_path) as img:
                                    img_width, img_height = img.size
#                                     print(f"  图像真实尺寸: {img_width}x{img_height}")
                                
                                # 优先将两个mask都调整到图像的真实尺寸
                                target_shape = (img_height, img_width)
                                # print(f"  将两个mask统一调整到图像尺寸: {target_shape}")
                                
                                import cv2
                                # 调整预测mask
                                if pred_mask.shape != target_shape:
                                    pred_mask = cv2.resize(
                                        pred_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                    # print(f"   pred_mask调整为: {pred_mask.shape}")
                                
                                # 调整真实mask 
                                if gt_mask.shape != target_shape:
                                    gt_mask = cv2.resize(
                                        gt_mask.astype(np.uint8), 
                                        (target_shape[1], target_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
#                                     print(f"    gt_mask调整为: {gt_mask.shape}")
                            
                            intersection, union = compute_iou(pred_mask, gt_mask)
                        except Exception as e:
                            print(f"警告: mask计算IoU失败 - 错误: {e}")
                            intersection, union = 0, 0
                        
                        # print(f" IoU计算结果: intersection={intersection}, union={union}")
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
                        # 复用已经加载的图像，避免重复读取
                        # current_image 已在上面加载过了
                        
                        # 获取边界框和掩码数据
                        pred_bbox = bboxes[0] if bboxes else None
                        gt_bbox = turn["bboxes"][0] if "bboxes" in turn and turn["bboxes"] else None
                        
                        # 构造文件名
                        vis_filename = f"image_{item['image_id']}_turn_{turn_idx}_ann_{turn['ann_ids'][0] if 'ann_ids' in turn and turn['ann_ids'] else 'unknown'}.png"
                        vis_path = os.path.join(args.vis_output_path, vis_filename)
                        
                        # 保存可视化
                        save_visualization(
                            image=current_image,
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
