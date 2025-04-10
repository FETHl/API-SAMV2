import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

class CRFPostProcessor:
    """
    Post-processes segmentation masks using Conditional Random Fields to refine boundaries.
    """
    def __init__(self, 
                 bilateral_sxy=80,
                 bilateral_srgb=13, 
                 iter_steps=5):
        """
        Initialize the CRF post-processor.
        
        Args:
            bilateral_sxy: Spatial sigma for the bilateral filter
            bilateral_srgb: Color sigma for the bilateral filter
            iter_steps: Number of CRF inference steps to perform
        """
        self.bilateral_sxy = bilateral_sxy
        self.bilateral_srgb = bilateral_srgb
        self.iter_steps = iter_steps
        
    def refine_mask(self, image, mask):
        """
        Refine a binary mask using CRF.
        
        Args:
            image: Original RGB image (NumPy array)
            mask: Binary mask (NumPy array with values 0 or 1)
            
        Returns:
            refined_mask: Improved binary mask with better boundaries
        """
        # Ensure image is in RGB format
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Prepare the input for CRF
        h, w = mask.shape[:2]
        
        # Get binary mask
        binary_mask = mask.astype(np.uint8)
        
        # Create labels for CRF (0 for background, 1 for foreground)
        labels = binary_mask.flatten().astype(np.int32)
        
        # Create unary potentials from the given labels
        n_labels = 2
        unary = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        
        # Setup the CRF
        d = dcrf.DenseCRF2D(w, h, n_labels)
        d.setUnaryEnergy(unary)
        
        # Add pairwise potentials (appearance and smoothness)
        pairwise_energy = create_pairwise_bilateral(
            sdims=(self.bilateral_sxy, self.bilateral_sxy),
            schan=(self.bilateral_srgb, self.bilateral_srgb, self.bilateral_srgb),
            img=rgb_image,
            chdim=2
        )
        d.addPairwiseEnergy(pairwise_energy, compat=10)
        
        # Add additional Gaussian pairwise potential
        d.addPairwiseGaussian(sxy=3, compat=3)
        
        # Perform inference
        q = d.inference(self.iter_steps)
        
        # Get the refined mask
        refined_mask = np.argmax(q, axis=0).reshape((h, w))
        
        return refined_mask.astype(np.uint8)