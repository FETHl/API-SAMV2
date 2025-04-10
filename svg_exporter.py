#!/usr/bin/env python3
"""
SVG exporter for segmentation results with color support
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import traceback

class SVGExporter:
    """
    SVG exporter for segmentation results with color support
    """
    def __init__(self, line_thickness=1.5):
        """Initialize the SVG exporter"""
        self.line_thickness = line_thickness
    
    def create_svg_document(self, size, metadata=None):
        """Create a new SVG document with specified size"""
        width, height = size
        
        # Create SVG root
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        svg.set('version', '1.1')
        
        # Add metadata
        if metadata:
            title = metadata.get('title', 'Segmentation with SAM')
            author = metadata.get('author', 'FETHl')
            date = metadata.get('date', '2025-03-14 09:39:04')
            version = metadata.get('version', '5.0.1')
            
            desc = ET.SubElement(svg, 'desc')
            desc.text = f"{title}\nCreated by: {author}\nDate: {date}\nVersion: {version}"
        
        return svg
    
    def add_shape_to_svg(self, svg_root, element, color=None, segment_id=None):
        """Add a geometric element to the SVG document with color"""
        element_type = element['type'].lower()
        shape_elem = None
        
        # Convert color to CSS format if provided
        color_css = None
        if color:
            r, g, b = color
            color_css = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        
        # Create SVG element based on type
        if element_type == 'cercle':
            center = element['center']
            radius = element['radius']
            
            shape_elem = ET.SubElement(svg_root, 'circle')
            shape_elem.set('cx', str(center[0]))
            shape_elem.set('cy', str(center[1]))
            shape_elem.set('r', str(radius))
            
        elif element_type == 'ellipse':
            center = element['center']
            axes = element['axes']
            angle = element.get('angle', 0)
            
            shape_elem = ET.SubElement(svg_root, 'ellipse')
            shape_elem.set('cx', str(center[0]))
            shape_elem.set('cy', str(center[1]))
            shape_elem.set('rx', str(axes[0]/2))
            shape_elem.set('ry', str(axes[1]/2))
            
            # Apply rotation if needed
            if angle != 0:
                shape_elem.set('transform', f'rotate({angle} {center[0]} {center[1]})')
                
        elif element_type == 'rectangle' or element_type == 'polygone' or element_type == 'contour':
            points = element['points']
            
            # Create path string
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"
            path_data += " Z"  # Close the path
            
            shape_elem = ET.SubElement(svg_root, 'path')
            shape_elem.set('d', path_data)
            
        elif element_type == 'lignes':
            segments = element['segments']
            
            # Create separate paths for each segment
            for i, (start, end) in enumerate(segments):
                line_elem = ET.SubElement(svg_root, 'line')
                line_elem.set('x1', str(start[0]))
                line_elem.set('y1', str(start[1]))
                line_elem.set('x2', str(end[0]))
                line_elem.set('y2', str(end[1]))
                
                # Apply styling
                line_elem.set('stroke', color_css if color_css else 'black')
                line_elem.set('stroke-width', str(self.line_thickness))
                line_elem.set('fill', 'none')
                
                # Add ID if provided
                if segment_id is not None:
                    line_elem.set('id', f'segment_{segment_id}_line_{i}')
        
        # Apply common styling to shape element
        if shape_elem is not None:
            shape_elem.set('stroke', color_css if color_css else 'black')
            shape_elem.set('stroke-width', str(self.line_thickness))
            shape_elem.set('fill', 'none')
            
            # Add ID if provided
            if segment_id is not None:
                shape_elem.set('id', f'segment_{segment_id}')
    
    def export_to_svg(self, segments_data, output_path, size, metadata=None):
        """Export segments to SVG with color support"""
        print(f"Exporting to SVG: {output_path}")
        
        try:
            # Create SVG document
            svg_root = self.create_svg_document(size, metadata)
            
            # Process segments
            for idx, segment in enumerate(segments_data):
                # Get segment color if available, otherwise use default
                color = segment.get('color', (0, 0, 1))  # Default to blue
                segment_id = segment.get('id', idx + 1)  # Use index+1 if no ID
                
                # Process each geometric element
                for element in segment['geometric_elements']:
                    self.add_shape_to_svg(svg_root, element, color, segment_id)
            
            # Format the XML for readability
            rough_string = ET.tostring(svg_root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(pretty_xml)
                
            print(f"SVG file exported successfully with {len(segments_data)} objects")
            return True
            
        except Exception as e:
            print(f"Error exporting to SVG: {str(e)}")
            traceback.print_exc()
            return False