import torch
import numpy as np
import re
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from PIL import Image as PILImage
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
from .base_model import (
    BaseVisionModel,
    DetectionModel,
    SegmentationModel,
    CountingModel,
    QAModel
)

class QwenVLModel(BaseVisionModel, DetectionModel, SegmentationModel, CountingModel, QAModel):
    """
    QwenVL model implementing all task interfaces with custom prompts
    """
    
    def __init__(self, model_path='Qwen/Qwen2.5-VL-7B-Instruct', 
                 segmentation_model_path="facebook/sam2-hiera-large"):
        """
        Initialize the QwenVL model
        
        Args:
            model_path (str): Path to the model
        """
        self.model_path = model_path
        
        # Initialize model
        if 'Qwen2.5' in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize segmentation model
        self.segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

        # Task-specific prompts
        if 'Qwen2.5' in model_path:
            self.DETECTION_PROMPT = """
            Locate "{query}", report the bboxes coordinates in JSON format.
            """
        else:
            self.DETECTION_PROMPT = """
            Please identify "{query}" in the image. 
            Return a JSON array where each object is represented as:
            {{"bbox": [x1, y1, x2, y2], "label": label}}
            
            Your response should be formatted as:
            ```json
            [
                {{"bbox": [x1, y1, x2, y2], "label": label}},
                ...
            ]
            ```
            """
        self.COUNTING_PROMPT = """
        Locate "{query}", report the bboxes coordinates in JSON format.
        """

        self.QA_PROMPT = """{query}"""
    
    def extract_json_from_response(self, response_text):
        """
        Extract JSON content from model response using triple quotes
        
        Args:
            response_text (str): Model response text
            
        Returns:
            dict or list: Parsed JSON content
        """
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {json_str}")
                return None
        
        # Fallback: try to find any JSON-like structure
        try:
            # Look for arrays
            array_pattern = r'\[(.*?)\]'
            array_match = re.search(array_pattern, response_text, re.DOTALL)
            if array_match:
                return json.loads(f"[{array_match.group(1)}]")
            
            # Look for objects
            object_pattern = r'\{(.*?)\}'
            object_match = re.search(object_pattern, response_text, re.DOTALL)
            if object_match:
                return json.loads(f"{{{object_match.group(1)}}}")
        except:
            pass
            
        return None
    
    def generate_response(self, image, prompt):
        """
        Generate response from the model
        
        Args:
            image (PIL.Image): Input image
            prompt (str): Text prompt
            
        Returns:
            str: Model response
        """
        # Resize image while maintaining aspect ratio
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to('cuda')
        
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14

        return output_text[0], (input_height, input_width)
    
    def _determine_task_type(self, instruction):
        """
        Determine the task type based on the instruction
        
        Args:
            instruction (str): Text instruction or query
            
        Returns:
            str: Task type ("detection", "segmentation", "counting", or "qa")
        """
        instruction_lower = instruction.lower()
        
        if "how many" in instruction_lower:
            return "counting"
        elif any(word in instruction_lower for word in ["segment", "mask", "outline"]):
            return "segmentation"
        elif any(word in instruction_lower for word in ["find", "locate", "detect", "identify"]):
            return "detection"
        else:
            return "vqa"
    
    def process_single_image(self, image, instruction, return_task_type=False):
        """
        Process a single image with given instruction
        
        Args:
            image (PIL.Image): Input image
            instruction (str): Text instruction or query
            
        Returns:
            dict: Results dictionary
        """
        task_type = self._determine_task_type(instruction)
        
        

        if task_type == "detection":
            result = self.detect_objects(image, instruction)
        elif task_type == "segmentation":
            result = self.segment_objects(image, instruction)
        elif task_type == "counting":
            result = self.count_objects(image, instruction)
        else:  # Default to QA
            result = self.answer_question(image, instruction)
        
        if return_task_type:
            return result, task_type
        else:
            return result
    
    def process_batch(self, batch_images, batch_instructions):
        """
        Process a batch of images with given instructions
        
        Args:
            batch_images (list): List of PIL Images
            batch_instructions (list): List of text instructions or queries
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        for image, instruction in zip(batch_images, batch_instructions):
            result = self.process_single_image(image, instruction)
            results.append(result)
        return results
    
    def detect_objects(self, image, query):
        """
        Detect objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to detect
            
        Returns:
            dict: Results with bounding boxes and scores
        """
        prompt = self.DETECTION_PROMPT.format(query=query)
        response, scale_factors = self.generate_response(image, prompt)
        
        # Get original image dimensions
        img_height, img_width = image.size[1], image.size[0]
        
        json_data = self.extract_json_from_response(response)
        
        bboxes = []
        points = []
        scores = []
        
        if json_data and isinstance(json_data, list):
            for item in json_data:
                for key in ['bbox', 'bbox_2d']:
                    if key in item:
                        # For Qwen2-VL, convert from normalized coordinates (0-1000) to actual image coordinates
                        if 'Qwen2.5' not in self.model_path:
                            bbox = [
                                int(item[key][0] * img_width / 1000),
                                int(item[key][1] * img_height / 1000),
                                int(item[key][2] * img_width / 1000),
                                int(item[key][3] * img_height / 1000)
                            ]
                        else:
                            # Original scaling for Qwen2.5-VL
                            bbox = [
                                int(item[key][0]),
                                int(item[key][1]),
                                int(item[key][2]),
                                int(item[key][3])
                            ]
                        bboxes.append(bbox)
                
                for key in ['point', 'point_2d']:
                    if key in item:
                        # Similarly handle points
                        if 'Qwen2.5' not in self.model_path:
                            point = [
                                int(item[key][0] * img_width / 1000),
                                int(item[key][1] * img_height / 1000)
                            ]
                        else:
                            point = [
                                int(item[key][0]),
                                int(item[key][1])
                            ]
                        points.append(point)
                
                for key in ['score', 'score_2d']:
                    if key in item:
                        scores.append(item[key])

                if len(scores) == 0:
                    scores.append(0.0)
        
        return {
            "bboxes": bboxes,
            "points": points,
            "scores": scores,
            "thinking": "",
            "full_response": response,
            "json_data": json_data
        }
    
    def detect_objects_batch(self, images, queries):
        """
        Detect objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of detection results
        """
        results = []
        for image, query in zip(images, queries):
            result = self.detect_objects(image, query)
            results.append(result)
        return results
    
    
    def generate_masks(self, image, bboxes, points):
        """
        Generate segmentation masks for given image, bounding boxes and points
        
        Args:
            image (PIL.Image): Input image
            bboxes (list): List of bounding boxes
            points (list): List of points
            
        Returns:
            numpy.ndarray: Combined segmentation mask
        """
        img_height, img_width = image.height, image.width
        mask_all = np.zeros((img_height, img_width), dtype=bool)
        
        if not bboxes:
            return mask_all
        
        try:
            self.segmentation_model.set_image(image)
            if not points:
                points = []
            if len(points) != len(bboxes):
                points.extend([None] * (len(bboxes) - len(points)))
            
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = self.segmentation_model.predict(
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                mask = masks[0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)
                
            return mask_all
        except Exception as e:
            print(f"Error generating masks: {e}")
            return mask_all
    
    def segment_objects(self, image, query):
        """
        Segment objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to segment
            
        Returns:
            dict: Results with masks and bounding boxes
        """
        try:
            prompt = self.DETECTION_PROMPT.format(query=query)
            response, scale_factors = self.generate_response(image, prompt)
            img_height, img_width = image.size[1], image.size[0]
            x_factor, y_factor = (1, 1) if 'Qwen2.5' in self.model_path \
                else (img_width / 1000, img_height / 1000)
            
            json_data = self.extract_json_from_response(response)
            
            bboxes = []
            points = []
            
            if json_data and isinstance(json_data, list):
                for item in json_data:
                    for key in ['bbox', 'bbox_2d']:
                        if key in item and len(item[key]) == 4:
                            bbox = [
                                int(item[key][0] * x_factor),
                                int(item[key][1] * y_factor),
                                int(item[key][2] * x_factor),
                                int(item[key][3] * y_factor)
                            ]
                            bboxes.append(bbox)
                    
                    for key in ['point', 'point_2d']:
                        if key in item and len(item[key]) == 2:
                            point = [
                                int(item[key][0] * x_factor),
                                int(item[key][1] * y_factor)
                            ]
                            points.append(point)
            
            masks = self.generate_masks(image, bboxes, points)
            
            return {
                "masks": masks,
                "bboxes": bboxes,
                "points": points,
                "thinking": "",
                "full_response": response,
                "json_data": json_data
            }
        except Exception as e:
            raise
            print(f"Error in segmentation: {e}")
            return {
                "masks": np.zeros((image.height, image.width), dtype=bool),
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "json_data": None
            }
    
    def segment_objects_batch(self, images, queries):
        """
        Segment objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of segmentation results
        """
        results = []
        for image, query in zip(images, queries):
            result = self.segment_objects(image, query)
            results.append(result)
        return results
    
    def count_objects(self, image, query):
        """
        Count objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to count
            
        Returns:
            dict: Results with count and bounding boxes
        """
        try:
            prompt = self.COUNTING_PROMPT.format(query=query)
            response, scale_factors = self.generate_response(image, prompt)
            
            # Get original image dimensions
            img_height, img_width = image.size[1], image.size[0]
            x_factor, y_factor = (1, 1) if 'Qwen2.5' in self.model_path \
                else (img_width / 1000, img_height / 1000)
            
            json_data = self.extract_json_from_response(response)
            
            bboxes = []
            points = []
            
            if json_data and isinstance(json_data, list):
                for item in json_data:
                    for key in ['bbox', 'bbox_2d']:
                        if key in item:
                            bbox = [
                                int(item[key][0] * x_factor),
                                int(item[key][1] * y_factor),
                                int(item[key][2] * x_factor),
                                int(item[key][3] * y_factor)
                            ]
                            bboxes.append(bbox)
                    
                    for key in ['point', 'point_2d']:
                        if key in item:
                            point = [
                                int(item[key][0] * x_factor),
                                int(item[key][1] * y_factor)
                            ]
                            points.append(point)
                
                return {
                    "count": len(bboxes),
                    "bboxes": bboxes,
                    "thinking": "",
                    "points": points,
                    "full_response": response,
                    "json_data": json_data
                }
            
            # If JSON extraction fails, extract count from text using regex
            # Match digits, number words and Roman numerals
            count = 0
            
            # Match cardinal numbers (e.g. "5", "five", "10")
            number_words = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 
                'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 
                'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
            }
            
            # Look for phrases like "there are X" or "I count X" or "X objects"
            count_patterns = [
                r'there (?:are|is) (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
                r'i (?:can |)(?:see|count|find) (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
                r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty) (?:objects|items)',
                r'count(?:ing)? (?:is|of) (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
                r'total (?:of|is) (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
                r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty) in total',
                r'found (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)',
                # Roman numerals
                r'there (?:are|is) (I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)',
                r'count(?:ing)? (?:is|of) (I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)',
                r'total (?:of|is) (I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)',
            ]
            
            roman_to_int = {
                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 
                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
                'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
                'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
            }
            
            import re
            response_lower = response.lower()
            
            for pattern in count_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    match_text = match.group(1)
                    if match_text.isdigit():
                        count = int(match_text)
                        break
                    elif match_text in number_words:
                        count = number_words[match_text]
                        break
                    elif match_text.upper() in roman_to_int:
                        count = roman_to_int[match_text.upper()]
                        break
            
            return {
                "count": count,
                "bboxes": bboxes,
                "thinking": "Extracted count from text response",
                "points": points,
                "full_response": response,
                "json_data": json_data
            }
            
        except Exception as e:
            print(f"Error in counting: {e}")
            return {
                "count": 0,
                "bboxes": [],
                "thinking": "",
                "points": [],
                "full_response": "",
                "json_data": None
            }
        
    
    def count_objects_batch(self, images, queries):
        """
        Count objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of counting results
        """
        results = []
        for image, query in zip(images, queries):
            result = self.count_objects(image, query)
            results.append(result)
        return results
    
    def answer_question(self, image, question):
        """
        Answer a question about an image
        
        Args:
            image: Input image
            question: Text question
            
        Returns:
            dict: Results with answer
        """
        try:
            prompt = self.QA_PROMPT.format(query=question)
            response, _ = self.generate_response(image, prompt)
            
            answer = response
            
            return {
                "answer": answer,
                "thinking": "",
                "full_response": response,
                "json_data": None
            }
        except Exception as e:
            print(f"Error in QA: {e}")
            return {
                "answer": "",
                "thinking": "",
                "full_response": "",
                "json_data": None
            }
    
    def answer_questions_batch(self, images, questions):
        """
        Answer questions about a batch of images
        
        Args:
            images: List of input images
            questions: List of text questions
            
        Returns:
            list: List of QA results
        """
        results = []
        for image, question in zip(images, questions):
            result = self.answer_question(image, question)
            results.append(result)
        return results