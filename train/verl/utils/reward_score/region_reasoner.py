import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def vision_reasoner_format_reward(predict_str: str, problem: str = None) -> float:
    def scene_format_reward(predict_str: str) -> float:
        scene_match = re.search(r'<scene>\s*(.*?)\s*</scene>', predict_str, re.DOTALL)
        if not scene_match:
            return 0.0
            
        scene_content = scene_match.group(1).strip()
        
        if len(scene_content) < 10:
            return 0.2  # 
        
        scene_keywords = ['scene', 'image', 'picture', 'photo', 'shows', 'contains', 'visible', 'appears', 'located']
        scene_lower = scene_content.lower()
        
        base_reward = 0.3
        if any(keyword in scene_lower for keyword in scene_keywords):
            base_reward = 0.5
            
        return base_reward
    
    def focus_format_reward(predict_str: str, problem: str = None) -> float:
        focus_match = re.search(r'<focus>\s*(.*?)\s*</focus>', predict_str, re.DOTALL)

        has_reference_bbox = False
        if problem:
            bbox_pattern = r'bbox\s*=\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
            has_reference_bbox = bool(re.search(bbox_pattern, problem.lower()))

        if has_reference_bbox:
            if not focus_match:
                return 0.0

            focus_content = focus_match.group(1).strip()
            if not focus_content:
                return 0.0

            focus_tokens_list = focus_content.split()
            word_count = len(focus_tokens_list)

            if word_count < 5:
                base_reward = 0.2  # 
            else:
                base_reward = 0.8  #

            if problem:
                problem_lower = problem.lower()
                focus_lower = focus_content.lower()

                problem_tokens = set(problem_lower.split())
                focus_tokens = set(focus_lower.split())
                overlap_ratio = len(problem_tokens & focus_tokens) / (len(problem_tokens) + 1e-6)

                if overlap_ratio > 0.6:  # 
                    base_reward = 0.0

            return base_reward

        if focus_match:
            return 0.0
        else:
            return 0.5 # 
    
    
    def think_format_reward(predict_str: str) -> float:
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', predict_str, re.DOTALL)
        if not think_match:
            return 0.0
            
        think_content = think_match.group(1).strip()
        
        if len(think_content) < 10:
            return 0.2
        
        return 0.5
    
    def answer_format_reward(predict_str: str) -> float:
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not answer_match:
            return 0.0
            
        answer_content = answer_match.group(1).strip()
        
        if len(answer_content) < 3:
            return 0.1
        
        return 0.5
    

    scene_reward = scene_format_reward(predict_str)
    focus_reward = focus_format_reward(predict_str, problem)
    think_reward = think_format_reward(predict_str)
    answer_reward = answer_format_reward(predict_str)
    
    def thinking_bbox_reward(predict_str: str, problem: str) -> float:
        thinking_bbox_reward = 0.0
        try:
            think_match = re.search(r'<think>\s*(.*?)\s*</think>', predict_str, re.DOTALL)
            if not think_match:
                return thinking_bbox_reward
                
            think_content = think_match.group(1).lower()
            problem_lower = problem.lower()
            
            problem_bbox_pattern = r'bbox\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            problem_bboxes = re.findall(problem_bbox_pattern, problem_lower)
            
            if not problem_bboxes:

                return 1.0
            
            think_bbox_pattern = r'bbox\s*=?\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            think_bboxes = re.findall(think_bbox_pattern, think_content)
            
            bbox_keywords = ['bbox', 'bounding box', 'bbox=', 'coordinates', 'bounding box=']
            if any(keyword in think_content for keyword in bbox_keywords):
                thinking_bbox_reward += 1.0 #
            
            if think_bboxes:
                problem_bbox_sets = set()
                for bbox in problem_bboxes:
                    problem_bbox_sets.add(tuple(map(int, bbox)))
                
                think_bbox_sets = set()
                for bbox in think_bboxes:
                    think_bbox_sets.add(tuple(map(int, bbox)))
                
                if think_bbox_sets:
                    consistent_bboxes = think_bbox_sets.intersection(problem_bbox_sets)
                    hallucinated_bboxes = think_bbox_sets - problem_bbox_sets
                    
                    if consistent_bboxes:
                        thinking_bbox_reward += 1.0 * (len(consistent_bboxes) / len(think_bbox_sets))
                    
                    if hallucinated_bboxes:
                        thinking_bbox_reward *= 0.5  
                        
        except Exception:
            pass
        return thinking_bbox_reward # 
    
    def segmentation_format(predict_str: str) -> float: 
        segmentation_format_reward = 0.0
        try: # 
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    thinking_bbox_reward_score = thinking_bbox_reward(predict_str, problem) #
    final_format_reward = scene_reward + focus_reward + think_reward + answer_reward

    reasoner_format_reward = final_format_reward + segmentation_format_reward + thinking_bbox_reward_score
    
    return reasoner_format_reward, final_format_reward

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120 
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]
            
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]
            
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)
            
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            iou_reward = (iou_matrix > 0.5).astype(float) #
            bbox_l1_reward = (l1_matrix < 10).astype(float) # 
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float) #
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward) 
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward
def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  #
    try:
        sentences = predict_str.split('.')
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0 # 
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward



def region_reasoner_compute_score(predict_str: str, ground_truth: str, problem: str = None, 
                                 text_only_inference_func=None, enable_consistency_reward=True, 
                                 model_inference_func=None) -> float:
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    


    format_reward, reasoner_format_reward = vision_reasoner_format_reward(predict_str, problem) #  
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth) # 
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str) # 
    

    
    consistency_reward = 0.0
    if enable_consistency_reward and problem:
        consistency_reward = vision_reasoner_consistency_reward(predict_str, problem, text_only_inference_func, model_inference_func)
        
    
    reward = format_reward + accuracy_reward + non_repeat_reward + consistency_reward

    return reward


def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # 
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # 
    xA = np.maximum(x11, np.transpose(x21)) # 
    yA = np.maximum(y11, np.transpose(y21)) # 
    xB = np.minimum(x12, np.transpose(x22)) # 
    yB = np.minimum(y12, np.transpose(y22)) # 
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # 
    return iou
def batch_l1_distance(boxes1, boxes2):
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N) 

def batch_points_distance(points1, points2): 
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):

    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

def vision_reasoner_consistency_reward(predict_str: str, problem: str, text_only_inference_func=None, model_inference_func=None) -> float:
  
    consistency_reward = 0.0
    
    try:
        scene_match = re.search(r'<scene>\s*(.*?)\s*</scene>', predict_str, re.DOTALL)
        focus_match = re.search(r'<focus>\s*(.*?)\s*</focus>', predict_str, re.DOTALL)
        original_answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        
        if not scene_match or not original_answer_match:
            return consistency_reward
            
        scene_content = scene_match.group(1).strip()
        focus_content = focus_match.group(1).strip() if focus_match else ""
        original_answer = original_answer_match.group(1).strip()
        
        question_match = re.search(r'find\s+"([^"]+)"', problem.lower())
        target_object = question_match.group(1) if question_match else "target objects"
        
        consistency_reward = simple_consistency_check(predict_str, scene_content, focus_content)

    except Exception as e:
        pass
        
    return consistency_reward  # 
def compare_answers(answer1: str, answer2: str) -> float:

    try:
        def extract_json(text):
            try:
                return json.loads(text.strip())
            except:
                pass
            
            json_pattern = r'\[.*?\]'
            json_match = re.search(json_pattern, text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            json_pattern = r'\{.*?\}'
            json_matches = re.findall(json_pattern, text, re.DOTALL)
            if json_matches:
                return [json.loads(match) for match in json_matches]
                
            return None
        
        data1 = extract_json(answer1)
        data2 = extract_json(answer2)
        
        if not data1 or not data2:
            return 0.0
            
        if not isinstance(data1, list):
            data1 = [data1]
        if not isinstance(data2, list):
            data2 = [data2]
            
        if len(data1) == 0 or len(data2) == 0:
            return 0.0
            
        count_consistency = 1.0 - min(abs(len(data1) - len(data2)) / max(len(data1), len(data2)), 0.5)
        
        similarity_matrix = np.zeros((len(data1), len(data2)))
        
        for i, item1 in enumerate(data1):
            for j, item2 in enumerate(data2):
                similarity_matrix[i][j] = calculate_item_similarity(item1, item2)
        
        if similarity_matrix.size > 0:
            cost_matrix = 1.0 - similarity_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matched_similarities = similarity_matrix[row_indices, col_indices]
            avg_similarity = np.mean(matched_similarities)
        else:
            avg_similarity = 0.0
        
        final_score = avg_similarity * count_consistency
        
        return min(final_score, 1.0)
        
    except Exception as e:
        return 0.0

def calculate_item_similarity(item1: dict, item2: dict) -> float:

    similarity = 0.0
    total_weight = 0.0
    
    if 'bbox_2d' in item1 and 'bbox_2d' in item2:
        bbox1 = np.array(item1['bbox_2d'])
        bbox2 = np.array(item2['bbox_2d'])
        
        iou = calculate_iou(bbox1, bbox2)
        
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        center_distance = np.linalg.norm(center1 - center2)
        center_similarity = max(0, 1.0 - center_distance / 200.0)  # 
        
        bbox_similarity = iou * 0.8 + center_similarity * 0.2
        similarity += bbox_similarity * 0.7  # 
        total_weight += 0.7
        
    if 'point_2d' in item1 and 'point_2d' in item2:
        point1 = np.array(item1['point_2d'])
        point2 = np.array(item2['point_2d'])
        
        distance = np.linalg.norm(point1 - point2)
        point_similarity = max(0, 1.0 - distance / 150.0)  # 
        similarity += point_similarity * 0.3  # 
        total_weight += 0.3
    
    if total_weight == 0:
        return 0.0
        
    return similarity

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def simple_consistency_check(predict_str: str, scene_content: str, focus_content: str) -> float:

    consistency_score = 0.0
    
    try:
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', predict_str, re.DOTALL)
        if not think_match:
            return consistency_score
            
        think_content = think_match.group(1).lower()
        scene_lower = scene_content.lower()
        focus_lower = focus_content.lower()
        
        scene_keywords = extract_keywords(scene_content)
        scene_references = sum(1 for keyword in scene_keywords if keyword in think_content)
        scene_consistency = min(scene_references / max(len(scene_keywords), 1), 1.0)
        
        focus_consistency = 1.0  #
        if focus_content.strip():
            focus_keywords = extract_keywords(focus_content)
            focus_references = sum(1 for keyword in focus_keywords if keyword in think_content)
            focus_consistency = min(focus_references / max(len(focus_keywords), 1), 1.0)
        
        logic_consistency = 0.0
        
        spatial_keywords = ['above', 'below', 'left', 'right', 'inside', 'overlapping', 'near', 'touching', 'next to']
        spatial_mentions = sum(1 for keyword in spatial_keywords if keyword in think_content)
        if spatial_mentions > 0:
            logic_consistency += 0.5
        
        comparison_keywords = ['compare', 'similar', 'different', 'match', 'closest', 'best', 'choose', 'select', 'find']
        comparison_mentions = sum(1 for keyword in comparison_keywords if keyword in think_content)
        if comparison_mentions > 0:
            logic_consistency += 0.3
            
        location_keywords = ['locate', 'identify', 'position', 'area', 'region', 'corner', 'center', 'coordinates']
        location_mentions = sum(1 for keyword in location_keywords if keyword in think_content)
        if location_mentions > 0:
            logic_consistency += 0.2
            
        reference_keywords = ['reference', 'bbox', 'guidance', 'based on', 'according to']
        reference_mentions = sum(1 for keyword in reference_keywords if keyword in think_content)
        if reference_mentions > 0:
            logic_consistency += 0.2
            

        consistency_score = scene_consistency * 1.0 + focus_consistency * 0.6 + logic_consistency * 0.4
        
    except Exception:
        pass
        
    return consistency_score

def extract_keywords(text: str) -> list:

    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    return keywords[:10]  # 

if __name__ == "__main__":
    predict_str = """
<answer>
[{"bbox_2d": [10, 100, 398, 423], "point_2d": [283, 169]}]
</answer>
"""
    ground_truth = """
[{"bbox_2d": [416, 7, 833, 553], "point_2d": [648, 249]}]"""
    print(predict_str)
    print(ground_truth)
    print(region_reasoner_compute_score(predict_str, ground_truth))

