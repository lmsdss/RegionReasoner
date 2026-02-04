import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class TaskRouter:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """Initialize task router"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.prompt_template = "Given a user instruction, please classify which type of task it is and output the final answer after \"####\".. The types are: 1) Segmentation/detection, 2) Counting, 3) Editing, 4) Caption/QA. The user instruction is: "
        
        self.labels = ['seg/det', 'count', 'editing', 'vqa']

    def route_task(self, instruction):
        """Route input instruction to corresponding task category
        
        Args:
            instruction: User input instruction
            
        Returns:
            dict: Dictionary containing predicted category and confidence
        """
        # Get model response
        response = self._get_model_response(instruction)
        
        # Extract category
        predicted_category = self._extract_category(response)
        
        return predicted_category
    
    def route_batch(self, instructions):
        """Batch route tasks
        
        Args:
            instructions: List of instructions
            
        Returns:
            list: List of result dictionaries
        """
        # Get batch responses
        responses = self._get_model_responses(instructions)
        
        results = []
        for instruction, response in zip(instructions, responses):
            category = self._extract_category(response)
            results.append(category)
            
        return results
    
    def _get_model_response(self, instruction):
        """Get model response for a single instruction"""
        return self._get_model_responses([instruction])[0]
    
    def _get_model_responses(self, instructions):
        """Get batch model responses
        
        Args:
            instructions: List of instructions
            
        Returns:
            list: List of responses
        """
        # Build batch messages
        message_batch = [
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": self.prompt_template + instruction}
            ]
            for instruction in instructions
        ]
        
        # Process batch template
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Batch tokenize
        model_inputs = self.tokenizer(
            text_batch, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            # Only keep newly generated tokens
            generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return responses
    
    def _extract_category(self, response):
        """Extract predicted category from model response"""
        response = response.lower()
        
        # 1. Find all number matches
        number_pattern = r'(?:type|category|task)?\s*(?:is|:)?\s*([1-4])'
        number_matches = re.findall(number_pattern, response)
        if number_matches:
            number = int(number_matches[-1])  # Take the last matched number
            if number == 1:
                return 'segmentation'
            elif number == 2:
                return 'counting'
            elif number == 3:
                return 'editing'
            elif number == 4:
                return 'vqa'
            
        # 2. Keyword matching - find all matching keywords
        matches = []
        if any(word in response for word in ['segmentation', 'detection', 'segment', 'detect', 'grounding']):
            matches.append('segmentation')
        if any(word in response for word in ['counting', 'count', 'number']):
            matches.append('counting')
        if any(word in response for word in ['editing', 'edit', 'modify']):
            matches.append('editing')
        if any(word in response for word in ['caption', 'qa', 'question', 'answer', 'describe']):
            matches.append('vqa')
            
        # Return the last matched category
        return matches[-1] if matches else "vqa"