import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch


def load_and_convert_BGR2RGB_image(image_path):
    """
    Charge une image à partir du chemin spécifié et convertit les couleurs de BGR à RGB.
    Stockée sous forme de tableau NumPy.
    cv2.imread(), une fonction d'OpenCV qui charge une image sous forme d'array NumPy.
    
    Args:
        image_path (str): Le chemin local de l'image à charger.

    Returns:
        image (numpy.ndarray): L'image chargée et convertie en RGB.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



