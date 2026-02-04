from abc import ABC, abstractmethod

class BaseVisionModel(ABC):
    """Abstract base class for vision models that process images and instructions"""
    
    @abstractmethod
    def process_single_image(self, image, instruction):
        """
        Process a single image and instruction
        
        Args:
            image: Input image
            instruction: Text instruction/query
            
        Returns:
            dict: Results dictionary
        """
        pass
    
    @abstractmethod
    def process_batch(self, batch_images, batch_instructions):
        """
        Process a batch of images and instructions
        
        Args:
            batch_images: List of input images
            batch_instructions: List of text instructions/queries
            
        Returns:
            list: List of result dictionaries
        """
        pass


class DetectionModel(ABC):
    """Interface for object detection tasks"""
    
    @abstractmethod
    def detect_objects(self, image, query):
        """
        Detect objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to detect
            
        Returns:
            dict: Results containing at least:
                - bboxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: List of confidence scores
                - thinking: Reasoning process (if available)
        """
        pass
    
    @abstractmethod
    def detect_objects_batch(self, images, queries):
        """
        Detect objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of result dictionaries
        """
        pass


class SegmentationModel(ABC):
    """Interface for segmentation tasks"""
    
    @abstractmethod
    def segment_objects(self, image, query):
        """
        Segment objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to segment
            
        Returns:
            dict: Results containing at least:
                - masks: Segmentation masks
                - bboxes: List of bounding boxes
                - thinking: Reasoning process (if available)
        """
        pass
    
    @abstractmethod
    def segment_objects_batch(self, images, queries):
        """
        Segment objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of result dictionaries
        """
        pass


class CountingModel(ABC):
    """Interface for counting tasks"""
    
    @abstractmethod
    def count_objects(self, image, query):
        """
        Count objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to count
            
        Returns:
            dict: Results containing at least:
                - count: Number of objects
                - bboxes: List of bounding boxes (optional)
                - thinking: Reasoning process (if available)
        """
        pass
    
    @abstractmethod
    def count_objects_batch(self, images, queries):
        """
        Count objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of result dictionaries
        """
        pass


class QAModel(ABC):
    """Interface for visual question answering tasks"""
    
    @abstractmethod
    def answer_question(self, image, question):
        """
        Answer a question about an image
        
        Args:
            image: Input image
            question: Text question
            
        Returns:
            dict: Results containing at least:
                - answer: Text answer
                - thinking: Reasoning process (if available)
        """
        pass
    
    @abstractmethod
    def answer_questions_batch(self, images, questions):
        """
        Answer questions about a batch of images
        
        Args:
            images: List of input images
            questions: List of text questions
            
        Returns:
            list: List of result dictionaries
        """
        pass