import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch


def draw_contours_mask_smooth_or_raw(mask, index=0, smooth=False, line_smoother=0.001, color=(0, 0, 0), line_width=3):
    """
    Dessine les contours en couleur d'un masque sur une image vide(fond noir).

    Args:
        mask (numpy.ndarray): Le masque binaire de l'image.
        index (int): L'indice du masque binaire à utiliser si `mask` est un ensemble de masques.
        smooth (bool): Indique si les contours doivent être lissés.
        line_smoother (float)(0.001 a 0.010): Le facteur de lissage des contours.
        color (tuple)(0 à 255, 0 à 255, 0 à 255): La couleur des contours en format RGB.
        line_width (int)(1 a 5): L'épaisseur des lignes de contour.

    Returns:
        contour_mask_for_fusion: le masque d'image avec les contours dessinés en couleur (au format RGB) utiliser pour la fusion.
        contour_mask: le masque d'image avec les contours utiliser pour l'affichage

    """
    try:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Le paramètre 'mask' doit être un tableau numpy.ndarray.")
        if not isinstance(index, int):
            raise TypeError("L'indice du masque doit être un entier.")
        if not (isinstance(index, int) and 0 <= index < len(mask)):
            raise ValueError("L'indice 'index' doit être un entier compris entre 0 et le nombre masque moins un.")
        if not isinstance(smooth, bool):
            raise TypeError("Le paramètre 'smooth' doit être de type booléen.")
        if not (0.001 <= line_smoother <= 0.01):
            raise ValueError("La valeur de line_smoother doit être comprise entre 0.001 et 0.01.")
        if not (isinstance(color, tuple) and len(color) == 3 and all(0 <= c <= 255 for c in color)):
            raise ValueError("Le paramètre 'color' doit être un tuple de trois entiers compris entre 0 et 255.")
        if not (1 <= line_width <= 5):
            raise ValueError("La valeur de 'line_width' doit être comprise entre 1 et 5.")

        # Convertir le masque booléen en masque binaire 8 bits
        mask_binary = mask[index]
        mask_array = (mask_binary > 0.5).astype(np.uint8) * 255

        # Trouver les contours dans le masque binaire
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer un masque vide pour dessiner les contours en couleur
        contour_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

        # Si smooth est True, approximer les contours pour rendre les lignes plus lisses
        if smooth:
            for contour in contours:
                epsilon = line_smoother * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                # Dessiner les contours approximés sur le masque vide

                # Changer la couleur des contours en BGR
                if all(color == 0 for color in color):
                    # Si Red=0, Green=0, et Blue=0, utiliser le blanc
                    cv2.drawContours(contour_mask, [smoothed_contour], -1, (255, 255, 255), line_width)
                else:
                    cv2.drawContours(contour_mask, [smoothed_contour], -1, color, line_width)

        else:
            # Changer la couleur des contours en BGR
            if all(color == 0 for color in color):
                # Si Red=0, Green=0, et Blue=0, utiliser le blanc
                cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), line_width)
            else:
                cv2.drawContours(contour_mask, contours, -1, color, line_width)
        # Créer une copie du masque pour la fusion
        contour_mask_for_fusion = contour_mask.copy()

        return contour_mask_for_fusion

    except TypeError as e:
        print("Erreur de type :", e)
        return None, None
    except ValueError as e:
        print("Erreur de valeur :", e)
        return None, None

def detect_edges_detail(image_path, mask,index=0,ksize=(0, 0),sigma=1, threshold1=1, threshold2=1, apertureSize=3, L2gradient=True,
                          dilation_enabled=True, kernel_width=5, kernel_height=5, color=(0, 0, 0)):
    """
    Détecte les bords (edges) dans une image en utilisant l'algorithme de Canny.

    Args:
        image_path (str): Le chemin vers l'image d'entrée.
        mask (numpy.ndarray): Le masque binaire de l'image.
        index (int): L'indice du masque binaire à utiliser si `mask` est un ensemble de masques.
        threshold1 (int)(0 à 255): Premier seuil pour le détecteur de Canny.
        threshold2 (int)(0 à 255): Deuxième seuil pour le détecteur de Canny.
        apertureSize (int)(3, 5, ou 7): Taille du noyau pour le filtre de Sobel.Notez que 1 correspond à l'absence de filtre de Sobel
        
        L2gradient(bool): dans Canny signifie calculer les gradients avec la norme L2 plutôt que L1, améliorant la précision mais potentiellement ralentissant le calcul
        ksize (tuple)(width, height) : Taille du noyau pour le filtre Gaussien. 
                                       Les dimensions doivent être des entiers positifs et impairs
                                       Si (0, 0) est spécifié, les dimensions du noyau seront calculées automatiquement à partir de sigma.
                                       Une valeur plus élevée entraîne un flou plus important.
        sigma (int)(0 à 4): Écart-type du noyau Gaussien utilisé pour le filtre Gaussien.
                            Contrôle l'intensité du flou Gaussien dans les directions X et Y.
        dilation_enabled (bool): Indique si la dilatation des contours est activée.
        kernel_width (int)(1 a 15): Largeur du noyau pour la dilatation des contours.
                                    (une opération de traitement d'image qui étend les régions de contours dans une image.)
        kernel_height (int)(1 a 15): Hauteur du noyau pour la dilatation des contours.
                                    (une opération de traitement d'image qui étend les régions de contours dans une image.)
        color (tuple)(0 à 255, 0 à 255, 0 à 255): La couleur des contours en format RGB.

    Returns:
        contour_mask_detail_result (numpy.ndarray): L'image avec les contours détectés.
    """
    try:
        if not isinstance(image_path, str):
            raise TypeError("Le chemin de l'image doit être une chaîne de caractères.")
        if not os.path.isfile(image_path):
            raise ValueError("Le chemin spécifié ne pointe pas vers un fichier valide.")
        if not isinstance(mask, np.ndarray):
            raise TypeError("Le masque doit être un tableau numpy.ndarray.")
        if not isinstance(index, int):
            raise TypeError("L'indice du masque doit être un entier.")
        if not (isinstance(index, int) and 0 <= index < len(mask)):
            raise ValueError("L'indice 'index' doit être un entier compris entre 0 et le nombre masque moins un.")
        if not isinstance(threshold1, int) or not 0 <= threshold1 <= 255:
            raise ValueError("La valeur de threshold1 doit être un entier entre 0 et 255.")
        if not isinstance(threshold2, int) or not 0 <= threshold2 <= 255:
            raise ValueError("La valeur de threshold2 doit être un entier entre 0 et 255.")
        if apertureSize not in [3, 5, 7]:
            raise ValueError("La valeur de apertureSize doit être parmi 1, 3, 5 ou 7.")
        if not isinstance(ksize, tuple) or len(ksize) != 2 or not all(isinstance(i, int) and i >= 0 for i in ksize):
            raise ValueError("La taille du noyau doit être un tuple de deux entiers positifs.")
        if not (isinstance(sigma, int) and 0 <= sigma <= 4):
            raise ValueError("La valeur de sigma doit être un entier entre 0 et 4.")
        if not isinstance(dilation_enabled, bool):
            raise TypeError("Le paramètre dilation_enabled doit être un booléen.")
        if not (1 <= kernel_width <= 15 and 1 <= kernel_height <= 15):
            raise ValueError("Les valeurs de kernel_width et kernel_height doivent être entre 1 et 15.")
        if not isinstance(color, tuple) or len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError("La couleur doit être un tuple de trois entiers entre 0 et 255.")
        
        mask_binary = mask[index]
        # Charger l'image
        image_originale = cv2.imread(image_path)
        # Binariser le masque
        mask_array = (mask_binary > 0.7).astype(np.uint8) * 255
        
        # Appliquer le masque à l'image
        masked_image = cv2.bitwise_and(image_originale, image_originale, mask=mask_array)

        # Appliquer un filtre Gaussien pour lisser l'image
        blurred_image = cv2.GaussianBlur(masked_image, ksize, sigmaX=sigma, sigmaY=sigma)

        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

        # Appliquer la détection de contours (edges) avec Canny

        edge = cv2.Canny(image=gray_image, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

        # Dilater les contours si l'option est activée
        if dilation_enabled:
            kernel = np.ones((kernel_width, kernel_height), np.uint8)  # Taille du noyau pour la dilation
            edge = cv2.dilate(edge, kernel, iterations=1)

        # Convertir l'image avec les contours en format RGB
        edges_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

        # Changer la couleur des contours en BGR
        if all(color == 0 for color in color):
            # Si Red=0, Green=0, et Blue=0, utiliser le blanc
            edges_rgb[edge != 0] = [255, 255, 255]  
        else:
            edges_rgb[edge != 0] = color

        contour_mask_result = edges_rgb.copy()


        #contour_mask_detail_result = cv2.cvtColor(contour_mask_result, cv2.COLOR_BGR2RGB)
        ####

        # ##to save mask
        # save_filename="contours_result.png"
        # save_path = os.path.join("/home/appuser/Grounded-Segment-Anything/AVH", save_filename)
        # if save_path is not None:
        #     contour_mask=cv2.cvtColor(contour_mask_result, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(save_path, contour_mask)

        return contour_mask_result
    except TypeError as e:
        print("Erreur de type :", e)
        return None, None
    except ValueError as e:
        print("Erreur de valeur :", e)
        return None, None

def fusion_masks(list_mask):
    """
    Fusionne plusieurs masques RGB sans mélanger les couleurs.
    
    :param list_mask: Liste de masques RGB sous forme de tableaux numpy.

    :return: Image des masque fusionné.

    :raises ValueError: Si la liste des masques est vide ou si tous les masques n'ont pas les mêmes dimensions.
    """
    # Vérifie si la liste des masques est vide
    if not list_mask:
        raise ValueError("La liste des masques ne peut pas être vide.")
    # Récupère les dimensions du premier masque de la liste
    height, width = list_mask[0].shape[:2]
    # Initialise le masque global avec des zéros
    global_mask = np.zeros((height, width, 3), dtype=np.uint8)
     # Parcours tous les masques dans la liste
    for mask in list_mask:
        # Vérifie si les dimensions du masque sont identiques à celles du premier masque
        if mask.shape[:2] != (height, width):
            raise ValueError("Tous les masques doivent avoir les mêmes dimensions.")
        # Trouve les indices où le masque global est nul et les remplace par les valeurs du masque courant
        mask_indices = np.where((global_mask == [0, 0, 0]).all(axis=2))
        global_mask[mask_indices] = mask[mask_indices]
    # Crée une image à partir du masque global fusionné
    mask = Image.fromarray(global_mask)
    
    return mask