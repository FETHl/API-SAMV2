import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

class ImageEnhancer:
    """
    Enhanced image preprocessing to optimize input for SAM segmentation.
    Improves contrast, lighting, and edge definition before segmentation.
    """
    def __init__(self, 
                 contrast_factor=1.2, 
                 brightness_factor=1.1,
                 sharpness_factor=1.3,
                 apply_clahe=True,
                 denoise=True):
        """
        Initialize the image enhancer with customizable parameters.
        
        Args:
            contrast_factor: Factor to enhance contrast (1.0 is neutral)
            brightness_factor: Factor to adjust brightness (1.0 is neutral)
            sharpness_factor: Factor to enhance sharpness (1.0 is neutral)
            apply_clahe: Whether to apply CLAHE for local contrast enhancement
            denoise: Whether to apply noise reduction
        """
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpness_factor = sharpness_factor
        self.apply_clahe = apply_clahe
        self.denoise = denoise
        
    def enhance(self, image):
        """
        Apply a series of enhancements to prepare image for segmentation.
        
        Args:
            image: NumPy array in BGR or RGB format
            
        Returns:
            enhanced_image: NumPy array with optimized properties for segmentation
        """
        # Convert to PIL image for some enhancements
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (OpenCV default) and convert to RGB for PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image)
        
        # Apply PIL-based enhancements
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(self.contrast_factor)
        
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(self.brightness_factor)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(self.sharpness_factor)
        
        # Convert back to NumPy for OpenCV operations
        enhanced = np.array(pil_image)
        
        # Apply CLAHE if requested (improves local contrast)
        if self.apply_clahe:
            if len(enhanced.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                # Convert back to RGB
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply denoising if requested
        if self.denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, None, 10, 10, 7, 21)
        
        # Convert back to BGR format if that's what was provided
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
        return enhanced

class EdgeEnhancer:
    """
    Enhances edges in the image to improve segmentation boundary detection.
    """
    def __init__(self, 
                 edge_strength=0.3,
                 sigma=1.0,
                 method='sobel'):
        """
        Initialize the edge enhancer.
        
        Args:
            edge_strength: Strength of edge enhancement (0.0-1.0)
            sigma: Blur sigma for edge detection
            method: Edge detection method ('sobel', 'canny', or 'dog')
        """
        self.edge_strength = edge_strength
        self.sigma = sigma
        self.method = method.lower()
        
    def enhance(self, image):
        """
        Enhance edges in the image.
        
        Args:
            image: NumPy array
            
        Returns:
            edge_enhanced_image: NumPy array with enhanced edges
        """
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (0, 0), self.sigma)
        
        # Edge detection based on selected method
        if self.method == 'sobel':
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            # Normalize
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
        elif self.method == 'canny':
            edges = cv2.Canny(blurred, 50, 150)
            
        elif self.method == 'dog':
            # Difference of Gaussians
            blur1 = cv2.GaussianBlur(gray, (0, 0), self.sigma)
            blur2 = cv2.GaussianBlur(gray, (0, 0), self.sigma * 1.6)
            edges = cv2.subtract(blur1, blur2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
        # Create edge overlay on original image
        if len(image.shape) == 3:
            # Color image
            result = image.copy()
            # Create a mask with the edges
            edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # Adjust the original image based on edge_strength
            result = cv2.addWeighted(result, 1.0, edge_mask, self.edge_strength, 0)
        else:
            # Grayscale image
            result = cv2.addWeighted(gray, 1.0, edges, self.edge_strength, 0)
            
        return result