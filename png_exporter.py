"""
PNG exporter for the Advanced SAM Segmentation Tool with transparency support
"""

import cv2
import numpy as np
import os
import traceback

class PNGExporter:
    """
    PNG exporter for segmentation results with transparency support
    """
    def __init__(self, line_thickness=5):
        """Initialize the PNG exporter"""
        self.line_thickness = line_thickness
    
    def export_to_png(self, segments_data, output_path, background_image=None, metadata=None):
        """
        Export segments to PNG with optional transparency
        
        Args:
            segments_data: List of segment dictionaries
            output_path: Output file path
            background_image: Optional background image (if None, transparent background is used)
            metadata: Optional metadata dictionary
        
        Returns:
            success: Boolean indicating success or failure
        """
        try:
            # Determine canvas size
            if background_image is not None:
                height, width = background_image.shape[:2]
                # Create canvas with the background image
                canvas = background_image.copy()
            else:
                # Try to determine size from segments
                max_x = max_y = 0
                for segment in segments_data:
                    for element in segment.get('geometric_elements', []):
                        if 'points' in element:
                            points = np.array(element['points'])
                            max_x = max(max_x, np.max(points[:, 0]))
                            max_y = max(max_y, np.max(points[:, 1]))
                
                # Set minimum size or use determined size
                width = max(800, int(max_x + 100))
                height = max(600, int(max_y + 100))
                
                # Create transparent canvas
                canvas = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Draw all segments
            for segment in segments_data:
                # Get segment color (RGB 0-1)
                color = segment.get('color', (0, 0, 1))  # Default blue
                
                # Convert to BGR or BGRA for OpenCV (0-255)
                if background_image is None:
                    # With transparency
                    bgra_color = (
                        int(color[2] * 255),  # B
                        int(color[1] * 255),  # G
                        int(color[0] * 255),  # R
                        255  # A (fully opaque)
                    )
                else:
                    # Without transparency
                    bgr_color = (
                        int(color[2] * 255),  # B
                        int(color[1] * 255),  # G
                        int(color[0] * 255)   # R
                    )
                
                # Process each geometric element
                for element in segment.get('geometric_elements', []):
                    if 'points' in element:
                        points = np.array(element['points'], dtype=np.int32)
                        
                        # Draw filled contour with transparency if no background
                        if background_image is None:
                            # Create mask for this contour
                            mask = np.zeros((height, width), dtype=np.uint8)
                            cv2.fillPoly(mask, [points], 255)
                            
                            # Create a color overlay with 30% opacity
                            overlay = np.zeros((height, width, 4), dtype=np.uint8)
                            overlay[mask == 255] = (bgra_color[0], bgra_color[1], bgra_color[2], 76)  # 30% opacity
                            
                            # Blend overlay with canvas
                            alpha_overlay = overlay[:, :, 3] / 255.0
                            alpha_canvas = 1.0 - alpha_overlay
                            
                            for c in range(3):  # RGB channels
                                canvas[:, :, c] = (alpha_overlay * overlay[:, :, c] + 
                                                 alpha_canvas * canvas[:, :, c])
                        
                        # Draw contour lines
                        if background_image is None:
                            cv2.polylines(canvas, [points], True, bgra_color, self.line_thickness, cv2.LINE_AA)
                        else:
                            cv2.polylines(canvas, [points], True, bgr_color, self.line_thickness, cv2.LINE_AA)
            
            # Save the image
            if background_image is None:
                # Save with transparency
                cv2.imwrite(output_path, canvas)
            else:
                # Save without transparency
                cv2.imwrite(output_path, canvas)
            
            print(f"PNG file exported successfully with {len(segments_data)} objects")
            return True
            
        except Exception as e:
            print(f"Error exporting to PNG: {str(e)}")
            traceback.print_exc()
            return False