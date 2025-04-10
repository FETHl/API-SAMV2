import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch

def load_sam_model(sam_checkpoint = "sam_vit_h_4b8939.pth", model_type = "default", device="cuda"):
    """
    Charge un modèle SAM pré-entraîné à partir d'un fichier télécharger
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    Initialise un prédicteur de masque et un générateur de masque SAM

    Args:
        sam_checkpoint (str): Le chemin vers le fichier du modèle SAM.
        model_type (str): Le type de modèle SAM à charger
        device (str): L'appareil sur lequel charger le modèle

    Returns:
        sam (torch.nn.Module): Le modèle SAM chargé.
        mask_predictor (SamPredictor): Le prédicteur de masque SAM initialisé.
        mask_generator (SamAutomaticMaskGenerator): Le générateur de masque SAM initialisé.
    """
    # Charger le modèle SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Initialiser le prédicteur et le générateur de masque SAM
    mask_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    return mask_predictor, mask_generator