"""
Flask-based web interface for interactive segmentation and contour editing.
"""
from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import cv2
import json
import base64
from io import BytesIO
from PIL import Image
from enhanced_segmentation import EnhancedSAMSegmentation
from export_utils import export_png, export_svg

app = Flask(__name__)

# Initialize SAM
sam_model = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and initialize the model"""
    global sam_model
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    img_np = np.array(img)
    
    # Initialize model if not already done
    if sam_model is None:
        model_type = request.form.get('model_type', 'vit_h')
        checkpoint_path = request.form.get('checkpoint_path', 'sam_vit_h_4b8939.pth')
        device = request.form.get('device', 'cuda')
        sam_model = EnhancedSAMSegmentation(model_type, checkpoint_path, device)
    
    # Set the image for segmentation
    sam_model.set_image(img_np)
    
    # Save image temporarily
    temp_path = 'static/temp/uploaded_image.png'
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    img.save(temp_path)
    
    return jsonify({
        'message': 'Image uploaded successfully',
        'image_path': temp_path,
        'width': img.width,
        'height': img.height
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    """Perform segmentation based on points or box"""
    data = request.json
    prompt_type = data.get('type', 'points')
    
    if prompt_type == 'points':
        points = np.array(data.get('points', []))
        labels = np.array(data.get('labels', [1] * len(points)))
        masks, scores, contours = sam_model.segment_with_points(
            points, 
            labels, 
            multimask_output=data.get('multimask', True),
            apply_crf_refinement=data.get('apply_crf', True)
        )
    elif prompt_type == 'box':
        box = np.array(data.get('box', [0, 0, 100, 100]))
        masks, scores, contours = sam_model.segment_with_box(
            box,
            multimask_output=data.get('multimask', True),
            apply_crf_refinement=data.get('apply_crf', True)
        )
    
    # Convert masks to base64 for display
    masks_b64 = []
    for mask in masks:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buffered = BytesIO()
        mask_img.save(buffered, format="PNG")
        masks_b64.append(base64.b64encode(buffered.getvalue()).decode())
    
    # Convert contours to JSON serializable format
    serializable_contours = []
    for contour_list in contours:
        serializable_contour_list = []
        for contour in contour_list:
            serializable_contour = contour.tolist()
            serializable_contour_list.append(serializable_contour)
        serializable_contours.append(serializable_contour_list)
    
    return jsonify({
        'masks': masks_b64,
        'scores': scores.tolist(),
        'contours': serializable_contours
    })

@app.route('/export', methods=['POST'])
def export_result():
    """Export segmentation result as PNG or SVG"""
    data = request.json
    export_type = data.get('type', 'png')
    
    # Get the mask data from base64
    mask_data = data.get('mask_data')
    mask_bytes = base64.b64decode(mask_data.split(',')[1])
    mask_img = Image.open(BytesIO(mask_bytes))
    mask = np.array(mask_img) > 0
    
    # Get the contours
    contours = json.loads(data.get('contours', '[]'))
    contours_np = []
    for contour_list in contours:
        contour_list_np = []
        for contour in contour_list:
            contour_np = np.array(contour, dtype=np.int32)
            contour_list_np.append(contour_np)
        contours_np.append(contour_list_np)
    
    if export_type == 'png':
        # Get original image
        img_path = 'static/temp/uploaded_image.png'
        img = cv2.imread(img_path)
        
        # Export as PNG
        output_path = 'static/exports/result.png'
        export_png(img, mask, output_path, overlay=data.get('overlay', True))
        
        return jsonify({
            'message': 'Exported as PNG',
            'file_path': output_path
        })
    
    elif export_type == 'svg':
        # Get image shape
        img_path = 'static/temp/uploaded_image.png'
        img = cv2.imread(img_path)
        
        # Export as SVG
        output_path = 'static/exports/result.svg'
        export_svg(img.shape, contours_np, output_path, include_image=data.get('include_image', False), image_path=img_path)
        
        return jsonify({
            'message': 'Exported as SVG',
            'file_path': output_path
        })
    
    return jsonify({'error': 'Invalid export type'}), 400

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download the exported file"""
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('static/temp', exist_ok=True)
    os.makedirs('static/exports', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)