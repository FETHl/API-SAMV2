import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch

def predictor_mask(mask_predictor, image):
    """
    Prédit les masques à partir de l'image en utilisant le prédicteur de masque donné.
    
    Args:
        mask_predictor (SamPredictor): Le prédicteur de masque à utiliser.
        image (numpy.ndarray): L'image sur laquelle prédire les masques.

    Returns:
        None
    """
    # Générer les masques à partir de l'image
    mask_predictor.set_image(image)



def predict_masks_method(mask_predictor, input_point=None, input_label=None, input_box=None,multimask_output=False):
    """
    Prédire les masques en utilisant le modèle de prédiction de masques de sam.

    Args:
        mask_predictor: Le modèle de prédiction de masques.
        input_point (numpy.ndarray)([X,Y]): Les coordonnées des points d'entrée.
        input_label (numpy.ndarray)([0] ou [1]): Étiquettes des points (1 pour les points positifs "garder", 0 pour les points négatifs"enlever").
        input_box(numpy.ndarray)([x_min, y_min, x_max, y_max]): Boîte d'entrée pour la prédiction.
        multimask_output: Indicateur indiquant si la sortie doit contenir plusieurs masques.

    Returns:
        masks: Masques prédits.
        scores: Scores associés aux masques prédits.
    """
    # Appeler la fonction de prédiction du prédicteur
    masks, scores, _ = mask_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=multimask_output,
    )
    

    return masks, scores



