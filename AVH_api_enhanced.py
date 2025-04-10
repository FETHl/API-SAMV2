from AVH_sam_model import load_sam_model
from AVH_load_image import load_and_convert_BGR2RGB_image
from AVH_mask_predictor import predictor_mask, predict_masks_method
from AVH_mask_maker import detect_edges_detail, draw_contours_mask_smooth_or_raw
from AVH_image_enhancer import ImageEnhancer, EdgeEnhancer
from AVH_crf_refiner import CRFPostProcessor
from AVH_contour_editor import ContourEditor
from fix_json_serialization_updated import CustomJSONEncoder, NumpyJSONEncoder

import base64
import cv2
from fastapi import FastAPI, Path as FastAPIPath, UploadFile, File, Query, HTTPException, Body, Response, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import traceback
import json
import numpy as np
import os
from pathlib import Path 
from PIL import Image
import signal
import shutil
import subprocess
from typing import List, Optional, Dict, Any, Union, Tuple
import uvicorn
import uuid
import time
import logging
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API-SAM")

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: print all registered routes
    print("REGISTERED ROUTES:")
    for route in app.routes:
        if hasattr(route, "methods"):
            # Regular route with HTTP methods
            methods = ','.join(route.methods)
            print(f"  {route.path} [{methods}]")
        elif hasattr(route, "app"):
            # Mounted app or static files
            print(f"  {route.path} [MOUNTED]")
        else:
            # Other type of route
            print(f"  {route.path} [UNKNOWN]")
    
    yield  # This is where the app runs
    
    # Shutdown: add any cleanup code here if needed
    print("API-SAM shutting down")

app = FastAPI(
    title="Enhanced API-SAM",
    description="Advanced image segmentation API with SAM, preprocessing, and interactive editing",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SAM model
logger.info("Loading SAM model...")
mask_predictor, mask_generator = load_sam_model()
logger.info("SAM model loaded successfully")

# Initialize image enhancer and CRF post-processor
image_enhancer = ImageEnhancer()
edge_enhancer = EdgeEnhancer()
crf_processor = CRFPostProcessor()

# Define base paths
BASE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
if not os.path.exists(BASE_IMAGE_PATH):
    os.makedirs(BASE_IMAGE_PATH)

# Create editor sessions storage
editor_sessions = {}

# Enhanced session class
class EnhancedImageSession:
    def __init__(self):
        self.image_path = None
        self.original_image = None
        self.enhanced_image = None
        self.mask = None
        self.name = None
        self.last_file_number = None
        self.contour_editor = ContourEditor()
        self.created_at = time.time()
        self.last_activity = time.time()
        self.masks = []  # Store generated masks
        self.mask_paths = []  # Store paths to masks
        self.contour_hierarchies = []  # Store hierarchies for inner/outer contours
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        
    def add_mask(self, mask, mask_path, hierarchy=None):
        """Add a mask to the session with optional hierarchy information"""
        self.masks.append(mask)
        self.mask_paths.append(mask_path)
        self.contour_hierarchies.append(hierarchy)

# Sessions dictionary
sessions = {}

# Background task to clean up old sessions
def cleanup_old_sessions():
    """Remove sessions that haven't been active for more than 1 hour"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session in sessions.items():
        if current_time - session.last_activity > 3600:  # 1 hour
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        # Clean up files
        session_folder = Path(BASE_IMAGE_PATH) / session_id
        if session_folder.exists():
            shutil.rmtree(session_folder)
        # Remove session
        del sessions[session_id]
    
    logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")

# Utility function to enhance contours with both inner and outer boundaries
def extract_hierarchical_contours(mask, min_area=50):
    """Extract both inner and outer contours in a hierarchical structure"""
    # Get contours with hierarchy information
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Filter out tiny contours
    if hierarchy is not None:
        # Organize contours by hierarchy
        organized_contours = []
        hierarchy = hierarchy[0]  # Unwrap hierarchy
        
        # Process each contour
        for i, (contour, h) in enumerate(zip(contours, hierarchy)):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Get hierarchy info - [Next, Previous, First_Child, Parent]
            next_idx, prev_idx, child_idx, parent_idx = h
            
            # Store contour with hierarchy info
            organized_contours.append({
                'contour': contour,
                'area': area,
                'hierarchy': {
                    'next': next_idx,
                    'prev': prev_idx,
                    'child': child_idx,
                    'parent': parent_idx
                },
                'is_hole': parent_idx >= 0  # It's a hole if it has a parent
            })
        
        return organized_contours
    
    return [{'contour': c, 'area': cv2.contourArea(c), 'hierarchy': None, 'is_hole': False} 
            for c in contours if cv2.contourArea(c) >= min_area]

# Add static files for web UI
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

@app.get("/")
async def root():
    """Redirect to the web UI"""
    return {"message": "API-SAM v2 is running. Access the UI at /ui/"}

# Health check endpoints
@app.get("/health")
def health_check():
    """Health check endpoint to verify API is operational"""
    return {"status": "ok", "timestamp": time.time()}

@app.head("/health")
def health_check_head():
    """Health check endpoint for HEAD requests"""
    return {"status": "ok"}

# Version endpoints
@app.get("/version")
def get_version():
    """Get API version information"""
    return {
        "version": "2.0.0", 
        "api_version": "2.0.0",
        "sam_version": "1.0", 
        "user": "FETHl",
        "date": "2025-04-08 14:17:24"
    }

@app.head("/version")
def get_version_head():
    """Version endpoint for HEAD requests"""
    return Response(status_code=200)

@app.post("/stop-server/")
async def stop_server():
    try:
        os.kill(os.getpid(), signal.SIGINT)
        return {"message": "Server is shutting down."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping/")
async def ping():
    return {"message": "pong", "status": "ok"}

@app.post("/session/create")
async def create_session(background_tasks: BackgroundTasks):
    """Create a new image processing session with a unique ID"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = EnhancedImageSession()
    
    # Create session directory
    session_folder = Path(BASE_IMAGE_PATH) / session_id
    os.makedirs(session_folder, exist_ok=True)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_old_sessions)
    
    logger.info(f"Created new session: {session_id}")
    return {"session_id": session_id, "message": "Session created successfully"}

@app.post("/upload/{image_id}")
async def upload_file(image_id: str, file: UploadFile = File(...), 
                      enhance: bool = Query(True), 
                      edge_enhance: bool = Query(False)):
    """
    Upload an image and optionally enhance it before processing
    """
    # Verify session exists or create new one
    if image_id not in sessions:
        sessions[image_id] = EnhancedImageSession()
   
    session = sessions[image_id]
    session.update_activity()
    
    # Save directory for the image
    save_path = Path(BASE_IMAGE_PATH)
    image_folder = save_path / image_id
    
    # Create directories if they don't exist
    if not image_folder.exists():
        os.makedirs(image_folder)
    
    # Save original image
    session.image_path = image_folder / file.filename
    with open(session.image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Load image
    original_image = load_and_convert_BGR2RGB_image(str(session.image_path))
    session.original_image = original_image
    
    # Apply enhancements if requested
    if enhance:
        enhanced_image = image_enhancer.enhance(original_image)
        # Save enhanced image
        enhanced_path = image_folder / f"enhanced_{file.filename}"
        cv2.imwrite(str(enhanced_path), cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        session.enhanced_image = enhanced_image
    else:
        session.enhanced_image = original_image
        enhanced_path = session.image_path
    
    # Apply edge enhancement if requested
    edge_path = None
    if edge_enhance:
        edge_enhanced = edge_enhancer.enhance(session.enhanced_image)
        # Save edge-enhanced image
        edge_path = image_folder / f"edge_enhanced_{file.filename}"
        cv2.imwrite(str(edge_path), cv2.cvtColor(edge_enhanced, cv2.COLOR_RGB2BGR))
        session.enhanced_image = edge_enhanced
    
    # Initialize mask predictor with the enhanced image
    predictor_mask(mask_predictor, session.enhanced_image)
    
    logger.info(f"Uploaded image for session {image_id}: {file.filename}")
    return {"message": "Image uploaded and preprocessed successfully", 
            "original_path": str(session.image_path),
            "enhanced_path": str(enhanced_path) if enhance else None,
            "edge_enhanced_path": str(edge_path) if edge_enhance else None}

@app.get("/get_image/")
async def get_image(image_id: str = Query(...), image_name: str = Query(...)):
    """
    Retrieve an image from the server
    """
    image_path = Path(f"{BASE_IMAGE_PATH}/{image_id}/{image_name}")
    if not image_path.is_file():
        return {"error": f"Image '{image_name}' not found for ID '{image_id}' on the server"}
    
    # Update session activity if it exists
    if image_id in sessions:
        sessions[image_id].update_activity()
        
    return FileResponse(image_path)
##################################################################################
@app.get("/auto/segment/{session_id}")
async def auto_segment(
    session_id: str = FastAPIPath(..., description="Session ID"),
    apply_crf: bool = Query(True),
    quality_threshold: float = Query(0.8),
    adjust_for_size: bool = Query(True),
    points_per_side: int = Query(32),
    pred_iou_thresh: float = Query(0.88),
    stability_score_thresh: float = Query(0.95)
):
    """
    Enhanced automatic segmentation that properly integrates with SAM model
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if an image has been uploaded
        if session.enhanced_image is None:
            raise HTTPException(status_code=400, detail="No image has been uploaded to this session")
        
        # Create directories
        mask_dir = Path(BASE_IMAGE_PATH) / session_id / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        # Clear previous masks
        session.masks = []
        session.mask_paths = []
        
        # Log start of mask generation
        logger.info("Generating automatic masks with SAM")
        
        # Use SAM model for better mask generation
        image = session.enhanced_image.copy()
        height, width = image.shape[:2]
        
        # Configure mask generator with parameters
        if hasattr(mask_generator, 'points_per_side'):
            mask_generator.points_per_side = points_per_side
        if hasattr(mask_generator, 'pred_iou_thresh'):
            mask_generator.pred_iou_thresh = pred_iou_thresh
        if hasattr(mask_generator, 'stability_score_thresh'):
            mask_generator.stability_score_thresh = stability_score_thresh
        
        # Generate masks
        generated_masks = mask_generator.generate(image)
        logger.info(f"SAM generated {len(generated_masks)} initial masks")
        
        # Extract masks and filter by quality
        masks = []
        for mask_data in generated_masks:
            if mask_data['predicted_iou'] >= quality_threshold:
                mask = mask_data['segmentation'].astype(np.uint8) * 255
                masks.append(mask)
        
        logger.info(f"Filtered to {len(masks)} quality masks")
        logger.info("Extracting hierarchical contours (inner and outer)")
        
        # Prepare mask hierarchies data
        mask_hierarchies = []
        
        # Apply CRF refinement if requested
        if apply_crf:
            refined_masks = []
            for mask in masks:
                try:
                    refined_mask = crf_processor.refine_mask(image, mask)
                    refined_masks.append(refined_mask)
                except Exception as e:
                    logger.error(f"CRF refinement error: {str(e)}")
                    refined_masks.append(mask)  # Use original mask if refinement fails
            masks = refined_masks
        
        # Create visualization image
        vis_image = image.copy()
        
        # Process and save masks
        for i, mask in enumerate(masks):
            # Save mask
            mask_path = mask_dir / f"mask_{i}.png"
            cv2.imwrite(str(mask_path), mask)
            
            # Add to session
            session.add_mask(mask, str(mask_path))
            
            # Extract contours with hierarchy
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours to create hierarchy information
            hierarchy_info = []
            
            if contours and hierarchy is not None:
                hierarchy = hierarchy[0]  # Get the first hierarchy level
                
                for j, contour in enumerate(contours):
                    # Convert to Python types for JSON serialization
                    area = float(cv2.contourArea(contour))
                    parent_idx = int(hierarchy[j][3])
                    is_hole = bool(parent_idx != -1)  # Convert to Python bool
                    
                    # Add to hierarchy info
                    hierarchy_info.append({
                        'contour': contour,  # Will be converted later
                        'is_hole': is_hole,  # Python bool
                        'area': area,        # Python float
                        'parent': parent_idx # Python int
                    })
            
            # Add hierarchy info for this mask
            mask_hierarchies.append(hierarchy_info)
            
            # Add to visualization with random color
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Save visualization
        vis_dir = Path(BASE_IMAGE_PATH) / session_id / "visualization"
        vis_dir.mkdir(exist_ok=True)
        vis_path = vis_dir / "masks_visualization.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Save hierarchical data to JSON file
        json_dir = Path(BASE_IMAGE_PATH) / session_id / "json"
        json_dir.mkdir(exist_ok=True)
        output_path = json_dir / "hierarchy_data.json"
        
        # Use custom encoder for safe serialization
        with open(output_path, 'w') as f:
            # Create a sanitized version of the data for serialization
            sanitized_data = {
                'masks': [{
                    'index': int(i),
                    'path': str(session.mask_paths[i]),
                    'contours': [{
                        'points': info['contour'].reshape(-1, 2).tolist(),
                        'is_hole': bool(info['is_hole']),
                        'area': float(info['area']),
                        'parent': int(info['parent'])
                    } for info in hierarchy]
                } for i, hierarchy in enumerate(mask_hierarchies)]
            }
            
            # Use the custom encoder to handle any remaining NumPy types
            json.dump(sanitized_data, f, cls=NumpyJSONEncoder)
        
        return {
            "message": "Automatic segmentation completed successfully",
            "mask_count": len(masks),
            "mask_paths": [str(path) for path in session.mask_paths],
            "visualization": str(vis_path),
            "hierarchy_json": str(output_path),
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-09 07:30:57"
        }
    except Exception as e:
        logger.error(f"Error in auto segmentation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Professional segmentation error: {str(e)}")

@app.head("/auto/segment/{session_id}")
async def auto_segment_head(session_id: str = FastAPIPath(..., description="Session ID")):
    """HEAD endpoint for auto segment to support endpoint detection"""
    return Response(status_code=200)

@app.get("/auto/segment/status/{session_id}")
async def get_segmentation_status(
    session_id: str = FastAPIPath(..., description="Session ID")
):
    """
    Get the status of a segmentation job
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Return the segmentation status
        return {
            "status": "completed" if hasattr(session, "masks") and len(session.masks) > 0 else "pending",
            "mask_count": len(session.masks) if hasattr(session, "masks") else 0,
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-08 14:19:51"
        }
    except Exception as e:
        logger.error(f"Error getting segmentation status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval error: {str(e)}")

@app.post("/auto/segment/settings")
async def update_segmentation_settings(
    quality_threshold: float = Body(0.8, description="Quality threshold for mask selection"),
    points_per_side: int = Body(32, description="Points per side for mask generation"),
    pred_iou_thresh: float = Body(0.88, description="Prediction IoU threshold"),
    stability_score_thresh: float = Body(0.95, description="Stability score threshold"),
    apply_crf: bool = Body(True, description="Apply CRF refinement"),
    include_inner_contours: bool = Body(True, description="Include inner contours in results"),
    hierarchical_export: bool = Body(True, description="Export contours in hierarchical format")
):
    """
    Update global segmentation settings
    """
    try:
        # Store settings in a global config
        global segmentation_settings
        segmentation_settings = {
            "quality_threshold": quality_threshold,
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "apply_crf": apply_crf,
            "include_inner_contours": include_inner_contours,
            "hierarchical_export": hierarchical_export,
            "last_updated": time.time(),
            "user": "FETHl",
            "date": "2025-04-08 14:19:51"
        }
        
        logger.info(f"Updated segmentation settings: {segmentation_settings}")
        
        return {
            "message": "Segmentation settings updated successfully",
            "settings": segmentation_settings
        }
    except Exception as e:
        logger.error(f"Error updating segmentation settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Settings update error: {str(e)}")

@app.get("/auto/masks/batch/{session_id}")
async def get_masks_batch(
    session_id: str = FastAPIPath(..., description="Session ID"),
    start_index: int = Query(0, description="Starting index of masks to retrieve"),
    count: int = Query(10, description="Number of masks to retrieve"),
    format: str = Query("json", pattern="^(json|png-urls)$", description="Response format")
):
    """
    Retrieve a batch of masks for a session
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if masks exist
        if not hasattr(session, "masks") or len(session.masks) == 0:
            raise HTTPException(status_code=404, detail="No masks found in this session")
        
        # Get the requested masks
        end_index = min(start_index + count, len(session.masks))
        if start_index >= len(session.masks):
            raise HTTPException(status_code=400, detail=f"Start index {start_index} exceeds mask count {len(session.masks)}")
        
        requested_masks = session.masks[start_index:end_index]
        requested_paths = session.mask_paths[start_index:end_index] if hasattr(session, "mask_paths") else []
        
        # Return in the requested format
        if format == "json":
            # Convert masks to JSON-serializable format with contours
            masks_data = []
            for i, mask in enumerate(requested_masks):
                # Extract contours
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convert contours to points
                contour_data = []
                for contour in contours:
                    points = contour.reshape(-1, 2).tolist()
                    if len(points) >= 3:  # Only include valid contours
                        contour_data.append(points)
                
                masks_data.append({
                    "index": start_index + i,
                    "contours": contour_data,
                    "area": int(np.sum(mask > 0)),
                    "path": str(session.mask_paths[start_index + i]) if hasattr(session, "mask_paths") else None
                })
            
            return {
                "masks": masks_data,
                "total_count": len(session.masks),
                "start_index": start_index,
                "returned_count": len(requested_masks),
                "timestamp": time.time(),
                "user": "FETHl",
                "date": "2025-04-08 14:19:51"
            }
        else:  # png-urls
            # Return URLs to access the PNG images
            urls = []
            for i in range(start_index, end_index):
                urls.append(f"/get_mask/{session_id}?index={i}&format=png")
            
            return {
                "mask_urls": urls,
                "total_count": len(session.masks),
                "start_index": start_index,
                "returned_count": len(requested_masks)
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving masks batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch retrieval error: {str(e)}")
# Add this endpoint to your API to serve individual masks

@app.get("/auto/masks/{session_id}")
async def get_mask(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to retrieve"),
    format: str = Query("png", pattern="^(png|json)$", description="Response format")
):
    """
    Retrieve a specific mask from a session
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask
        mask = session.masks[index]
        
        # Return based on requested format
        if format.lower() == "png":
            # Create a temporary file to return
            output_dir = Path(BASE_IMAGE_PATH) / session_id / "temp"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"mask_{index}.png"
            
            # Save the mask
            cv2.imwrite(str(output_path), mask)
            
            return FileResponse(
                output_path,
                media_type="image/png",
                filename=f"mask_{index}.png"
            )
        else:  # JSON format
            # Find contours for the mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert contours to points
            contour_data = []
            
            # Process contours with hierarchy
            if contours and hierarchy is not None:
                hierarchy = hierarchy[0]  # Flatten hierarchy array
                
                for i, contour in enumerate(contours):
                    # Convert to Python types for JSON serialization
                    is_inner = int(hierarchy[i][3]) != -1  # Has parent
                    area = float(cv2.contourArea(contour))
                    points = contour.reshape(-1, 2).tolist()
                    
                    contour_data.append({
                        "points": points,
                        "is_inner": bool(is_inner),
                        "area": area,
                        "parent_idx": int(hierarchy[i][3])
                    })
            
            # Return JSON data
            return {
                "mask_index": index,
                "contours": contour_data,
                "timestamp": time.time(),
                "user": "FETHl",
                "date": "2025-04-09 08:03:15"
            }
    except Exception as e:
        logger.error(f"Error retrieving mask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving mask: {str(e)}")
    

@app.post("/auto/masks/filter/{session_id}")
async def filter_masks(
    session_id: str = FastAPIPath(..., description="Session ID"),
    min_area: int = Body(None, description="Minimum mask area"),
    max_area: int = Body(None, description="Maximum mask area"),
    min_complexity: float = Body(None, description="Minimum contour complexity"),
    max_complexity: float = Body(None, description="Maximum contour complexity"),
    keep_indices: List[int] = Body(None, description="Specific mask indices to keep")
):
    """
    Filter masks based on criteria and remove those that don't match
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if masks exist
        if not hasattr(session, "masks") or len(session.masks) == 0:
            raise HTTPException(status_code=404, detail="No masks found in this session")
        
        # Initialize list of indices to keep
        indices_to_keep = set()
        
        # If specific indices were provided, start with those
        if keep_indices:
            for idx in keep_indices:
                if 0 <= idx < len(session.masks):
                    indices_to_keep.add(idx)
        else:
            # Filter based on criteria
            for i, mask in enumerate(session.masks):
                should_keep = True
                
                # Check area constraints
                if min_area is not None or max_area is not None:
                    area = np.sum(mask > 0)
                    if min_area is not None and area < min_area:
                        should_keep = False
                    if max_area is not None and area > max_area:
                        should_keep = False
                
                # Check complexity constraints
                if (min_complexity is not None or max_complexity is not None) and should_keep:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Calculate perimeter and area for complexity
                        perimeter = cv2.arcLength(contours[0], True)
                        area = cv2.contourArea(contours[0])
                        
                        if area > 0:
                            # Complexity is a ratio of perimeter to area
                            complexity = perimeter**2 / (4 * np.pi * area)
                            
                            if min_complexity is not None and complexity < min_complexity:
                                should_keep = False
                            if max_complexity is not None and complexity > max_complexity:
                                should_keep = False
                
                if should_keep:
                    indices_to_keep.add(i)
        
        # Create a backup of the original masks
        original_masks = session.masks.copy()
        original_paths = session.mask_paths.copy() if hasattr(session, "mask_paths") else []
        
        # Filter the masks
        session.masks = [original_masks[i] for i in sorted(indices_to_keep)]
        if hasattr(session, "mask_paths") and session.mask_paths:
            session.mask_paths = [original_paths[i] for i in sorted(indices_to_keep)]
        
        logger.info(f"Filtered masks from {len(original_masks)} to {len(session.masks)}")
        
        return {
            "message": f"Successfully filtered masks from {len(original_masks)} to {len(session.masks)}",
            "kept_indices": sorted(indices_to_keep),
            "removed_count": len(original_masks) - len(session.masks),
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-08 14:19:51"
        }
    except Exception as e:
        logger.error(f"Error filtering masks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mask filtering error: {str(e)}")

# Hierarchical contour visualization endpoint
@app.get("/auto/visualize/hierarchy/{session_id}")
async def visualize_contour_hierarchy(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to visualize"),
    color_mode: str = Query("rainbow", pattern="^(rainbow|depth|groups)$", description="Coloring mode for contours")
):
    """
    Visualize contour hierarchy with color-coded parent-child relationships
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if not hasattr(session, "masks") or index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask
        mask = session.masks[index]
        
        # Get image dimensions from mask
        height, width = mask.shape[:2]
        
        # Create visualization image - white background
        vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Find contours with hierarchy information
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and hierarchy is not None:
            # Flatten hierarchy array
            hierarchy = hierarchy[0]
            
            # Function to generate colors
            def get_color(idx, mode):
                if mode == "rainbow":
                    # Generate rainbow colors based on index
                    hue = (idx * 30) % 180
                    return cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                elif mode == "depth":
                    # Calculate contour depth (distance from outermost)
                    depth = 0
                    parent = hierarchy[idx][3]
                    while parent != -1:
                        depth += 1
                        parent = hierarchy[parent][3]
                    
                    # Deeper contours are redder
                    intensity = min(255, 50 + depth * 50)
                    return [0, 0, intensity]
                elif mode == "groups":
                    # Group by immediate parent
                    parent = hierarchy[idx][3]
                    if parent == -1:
                        # Outermost contours are blue
                        return [255, 0, 0]
                    else:
                        # Child contours get unique colors based on parent
                        parent_hash = hash(parent) % 5
                        colors = [
                            [0, 255, 0],    # Green
                            [0, 0, 255],    # Red
                            [255, 255, 0],  # Cyan
                            [255, 0, 255],  # Magenta
                            [0, 255, 255]   # Yellow
                        ]
                        return colors[parent_hash]
            
            # Draw contours with appropriate colors
            for i, contour in enumerate(contours):
                color = get_color(i, color_mode)
                cv2.drawContours(vis_image, contours, i, color, 2)
                
                # Add contour index label
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis_image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
        
        # Save the visualization
        vis_dir = Path(BASE_IMAGE_PATH) / session_id / "visualization"
        vis_dir.mkdir(exist_ok=True)
        vis_path = vis_dir / f"hierarchy_visualization_{index}_{color_mode}.png"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Return the image
        return FileResponse(vis_path)
    except Exception as e:
        logger.error(f"Error visualizing contour hierarchy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

# Export contours with full hierarchy information
@app.get("/auto/export/hierarchy/{session_id}")
async def export_contour_hierarchy(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to export"),
    format: str = Query("json", pattern="^(json|svg)$", description="Export format")
):
    """
    Export contours with their hierarchical structure
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if not hasattr(session, "masks") or index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask
        mask = session.masks[index]
        
        # Find contours with hierarchy information
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        if format == "json":
            # Convert to hierarchical JSON structure
            contour_data = []
            
            if contours and hierarchy is not None:
                # Flatten hierarchy array
                hierarchy = hierarchy[0]
                
                # Build hierarchy tree
                def build_tree(idx):
                    contour = contours[idx]
                    contour_points = contour.reshape(-1, 2).tolist()
                    
                    # Basic contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    node = {
                        "index": idx,
                        "points": contour_points,
                        "area": area,
                        "perimeter": perimeter,
                        "complexity": perimeter**2 / (4 * np.pi * area) if area > 0 else 0,
                        "children": []
                    }
                    
                    # Find children
                    for i, h in enumerate(hierarchy):
                        if h[3] == idx:  # if parent is current idx
                            node["children"].append(build_tree(i))
                    
                    return node
                
                # Find root contours (no parent)
                for i, h in enumerate(hierarchy):
                    if h[3] == -1:  # No parent
                        contour_data.append(build_tree(i))
            
            # Save as JSON
            json_path = export_dir / f"hierarchy_export_{index}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "mask_index": index,
                    "contours": contour_data,
                    "timestamp": time.time(),
                    "user": "FETHl",
                    "date": "2025-04-08 14:19:51"
                }, f, indent=2)
            
            # Return the JSON data
            return FileResponse(
                json_path,
                media_type="application/json",
                filename=f"hierarchy_export_{index}.json"
            )
            
        elif format == "svg":
            # Get image dimensions from mask
            height, width = mask.shape[:2]
            
            # Create SVG with hierarchical structure
            svg_path = export_dir / f"hierarchy_export_{index}.svg"
            
            # Start SVG document
            svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description>
      <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Contour Hierarchy Export</dc:title>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">FETHl - API-SAM v2</dc:creator>
      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">2025-04-08 14:19:51</dc:date>
    </rdf:Description>
  </rdf:RDF>
</metadata>
"""
            
            # Include base64 encoded image if available
            if session.original_image is not None:
                # Convert image to bytes
                success, buffer = cv2.imencode(".png", cv2.cvtColor(session.original_image, cv2.COLOR_RGB2BGR))
                if success:
                    img_bytes = buffer.tobytes()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    svg_content += f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}" opacity="0.2" />\n'
            
            if contours and hierarchy is not None:
                # Flatten hierarchy array
                hierarchy = hierarchy[0]
                
                # Function to process contour and its children
                def process_contour_hierarchy(idx, depth=0):
                    nonlocal svg_content
                    contour = contours[idx]
                    
                    # Generate color based on depth
                    hue = (depth * 30) % 360
                    
                    # Convert contour to SVG path
                    path_data = "M "
                    for point in contour.reshape(-1, 2):
                        x, y = point
                        path_data += f"{x},{y} "
                    path_data += "Z"  # Close the path
                    
                    # Add contour as group with metadata
                    svg_content += f'<g id="contour-{idx}" class="depth-{depth}" data-parent="{hierarchy[idx][3]}">\n'
                    
                    # Add path with hierarchical styling
                    opacity = max(0.2, 1.0 - depth * 0.15)  # Decrease opacity with depth
                    stroke_width = max(0.5, 2.0 - depth * 0.3)  # Decrease stroke width with depth
                    
                    svg_content += f'  <path d="{path_data}" fill="none" stroke="hsl({hue}, 80%, 50%)" ' + \
                                  f'stroke-width="{stroke_width}" opacity="{opacity}" />\n'
                    
                    # Find and process children
                    for i, h in enumerate(hierarchy):
                        if h[3] == idx:  # if parent is current idx
                            process_contour_hierarchy(i, depth + 1)
                    
                    svg_content += '</g>\n'
                
                # Process all root contours
                for i, h in enumerate(hierarchy):
                    if h[3] == -1:  # No parent
                        process_contour_hierarchy(i)
            
            # Close SVG document
            svg_content += "</svg>"
            
            # Write SVG file
            with open(svg_path, "w") as f:
                f.write(svg_content)
            
            # Return SVG file
            return FileResponse(
                svg_path,
                media_type="image/svg+xml",
                filename=f"hierarchy_export_{index}.svg"
            )
    except Exception as e:
        logger.error(f"Error exporting contour hierarchy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")
    

# Enhanced contour extraction with inner contours
@app.get("/auto/contours/advanced/{session_id}")
async def get_advanced_contours(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to get contours for"),
    include_inner: bool = Query(True, description="Include inner contours"),
    simplify: bool = Query(False, description="Simplify contours for cleaner output"),
    tolerance: float = Query(1.0, description="Tolerance for simplification (higher = more simplified)"),
    min_area: int = Query(10, description="Minimum contour area to include"),
    hierarchical: bool = Query(True, description="Return contours in hierarchical structure")
):
    """
    Professional contour extraction with inner/outer contour detection
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask for contour extraction
        mask = session.masks[index]
        
        # Get contours with hierarchy information
        retrieval_mode = cv2.RETR_TREE if include_inner else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare the response structure
        if hierarchical and include_inner and hierarchy is not None:
            # Organize contours in hierarchical structure
            hierarchy = hierarchy[0]  # Flatten hierarchy array
            contour_tree = []
            
            # Build hierarchical structure for each root contour
            def build_contour_tree(idx):
                contour = contours[idx]
                area = cv2.contourArea(contour)
                
                # Skip small contours if requested
                if area < min_area:
                    return None
                
                # Simplify contour if requested
                if simplify and len(contour) > 10:
                    epsilon = tolerance * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list format
                points = contour.reshape(-1, 2).tolist()
                if len(points) < 3:  # Skip invalid contours
                    return None
                
                # Create node with contour data
                node = {
                    "points": points,
                    "area": float(area),
                    "perimeter": float(cv2.arcLength(contour, True)),
                    "is_inner": False,
                    "children": []
                }
                
                # Find and add children
                for i, h in enumerate(hierarchy):
                    if h[3] == idx:  # Parent is current index
                        child_node = build_contour_tree(i)
                        if child_node:
                            child_node["is_inner"] = True
                            node["children"].append(child_node)
                
                return node
            
            # Find root contours (no parent) and build tree
            for i, h in enumerate(hierarchy):
                if h[3] == -1:  # No parent
                    tree_node = build_contour_tree(i)
                    if tree_node:
                        contour_tree.append(tree_node)
            
            # Return hierarchical structure
            return {
                "mask_index": index,
                "hierarchical_contours": contour_tree,
                "contour_count": len(contour_tree),
                "timestamp": time.time(),
                "user": "FETHl",
                "date": "2025-04-08 14:23:14"
            }
        else:
            # Flat structure with just contour points
            contour_data = []
            for i, contour in enumerate(contours):
                # Check minimum area
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                # Simplify if requested
                if simplify and len(contour) > 10:
                    epsilon = tolerance * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list and append
                points = contour.reshape(-1, 2).tolist()
                if len(points) >= 3:  # Only include valid contours
                    is_inner = False
                    if include_inner and hierarchy is not None:
                        # Check if this is an inner contour (has a parent)
                        parent_idx = hierarchy[0][i][3]
                        is_inner = parent_idx != -1
                    
                    contour_data.append({
                        "points": points,
                        "area": float(area),
                        "is_inner": is_inner
                    })
            
            return {
                "contours": contour_data,
                "total_count": len(contour_data),
                "outer_count": sum(1 for c in contour_data if not c["is_inner"]),
                "inner_count": sum(1 for c in contour_data if c["is_inner"])
            }
    except Exception as e:
        logger.error(f"Error extracting advanced contours: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Contour extraction error: {str(e)}")

# Advanced mask post-processing endpoint
@app.post("/auto/postprocess/{session_id}")
async def postprocess_mask(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Body(..., description="Index of the mask to process"),
    operations: List[Dict] = Body(..., description="List of operations to apply"),
    save_as_new: bool = Body(False, description="Save as new mask instead of replacing")
):
    """
    Apply advanced post-processing operations to a mask
    
    Operations include:
    - morphology: {type: "erode"|"dilate"|"open"|"close", kernel_size: int, iterations: int}
    - smooth: {radius: int, sigma: float}
    - fill_holes: {min_size: int, max_size: int}
    - remove_small: {min_size: int}
    - edge_enhance: {sigma: float, strength: float}
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the original mask
        original_mask = session.masks[index].copy()
        processed_mask = original_mask.copy()
        
        # Track applied operations
        applied_operations = []
        
        # Apply each operation in sequence
        for op in operations:
            op_type = op.get("type", "").lower()
            
            if op_type in ["erode", "dilate", "open", "close"]:
                # Morphological operations
                kernel_size = op.get("kernel_size", 3)
                iterations = op.get("iterations", 1)
                
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                if op_type == "erode":
                    processed_mask = cv2.erode(processed_mask, kernel, iterations=iterations)
                elif op_type == "dilate":
                    processed_mask = cv2.dilate(processed_mask, kernel, iterations=iterations)
                elif op_type == "open":
                    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
                elif op_type == "close":
                    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                
                applied_operations.append(f"{op_type}(kernel={kernel_size}, iter={iterations})")
            
            elif op_type == "smooth":
                # Smoothing operation
                radius = op.get("radius", 5)
                sigma = op.get("sigma", 1.5)
                
                # Convert to float for processing
                temp = processed_mask.astype(np.float32) / 255.0
                
                # Apply Gaussian blur
                temp = cv2.GaussianBlur(temp, (radius*2+1, radius*2+1), sigma)
                
                # Threshold back to binary
                processed_mask = (temp > 0.5).astype(np.uint8) * 255
                
                applied_operations.append(f"smooth(radius={radius}, sigma={sigma})")
            
            elif op_type == "fill_holes":
                # Fill holes within the mask
                min_size = op.get("min_size", 0)
                max_size = op.get("max_size", float('inf'))
                
                # Invert mask to find holes
                inverted = cv2.bitwise_not(processed_mask)
                
                # Label connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)
                
                # Prepare new mask by inverting back
                filled_mask = cv2.bitwise_not(inverted)
                
                # Fill holes within size range
                for i in range(1, num_labels):  # Skip background (0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if min_size <= area <= max_size:
                        # This is a hole we want to fill
                        component_mask = (labels == i).astype(np.uint8) * 255
                        filled_mask = cv2.bitwise_or(filled_mask, component_mask)
                
                processed_mask = filled_mask
                applied_operations.append(f"fill_holes(min={min_size}, max={max_size})")
            
            elif op_type == "remove_small":
                # Remove small connected components
                min_size = op.get("min_size", 100)
                
                # Find connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)
                
                # Create new mask with only large components
                filtered_mask = np.zeros_like(processed_mask)
                
                for i in range(1, num_labels):  # Skip background (0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_size:
                        # Keep this component
                        component_mask = (labels == i).astype(np.uint8) * 255
                        filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
                
                processed_mask = filtered_mask
                applied_operations.append(f"remove_small(min={min_size})")
            
            elif op_type == "edge_enhance":
                # Enhance edges of the mask
                sigma = op.get("sigma", 1.0)
                strength = op.get("strength", 1.0)
                
                # Detect edges using Canny
                edges = cv2.Canny(processed_mask, 50, 150)
                
                # Dilate edges
                kernel_size = int(strength * 2) + 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Combine with original mask
                processed_mask = cv2.bitwise_or(processed_mask, dilated_edges)
                
                applied_operations.append(f"edge_enhance(sigma={sigma}, strength={strength})")
        
        # Save the processed mask
        if save_as_new:
            # Create a new mask entry
            mask_dir = Path(BASE_IMAGE_PATH) / session_id / "masks"
            mask_dir.mkdir(exist_ok=True)
            
            # Generate new mask number
            new_index = len(session.masks)
            mask_path = mask_dir / f"mask_{new_index}_processed.png"
            
            # Save the mask
            cv2.imwrite(str(mask_path), processed_mask)
            
            # Add to session
            session.add_mask(processed_mask, str(mask_path))
            
            return {
                "message": f"Created new processed mask at index {new_index}",
                "original_index": index,
                "new_index": new_index,
                "applied_operations": applied_operations,
                "timestamp": time.time(),
                "user": "FETHl",
                "date": "2025-04-08 14:23:14"
            }
        else:
            # Replace original mask
            # Get original mask path
            mask_path = session.mask_paths[index]
            
            # Save the mask
            cv2.imwrite(mask_path, processed_mask)
            
            # Update in session
            session.masks[index] = processed_mask
            
            return {
                "message": f"Updated mask at index {index}",
                "index": index,
                "applied_operations": applied_operations,
                "timestamp": time.time(),
                "user": "FETHl",
                "date": "2025-04-08 14:23:14"
            }
    except Exception as e:
        logger.error(f"Error in mask post-processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Post-processing error: {str(e)}")

# Professional contour analysis endpoint
@app.get("/auto/analyze/{session_id}")
async def analyze_segmentation(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(None, description="Specific mask index to analyze, None for all"),
    detailed: bool = Query(False, description="Include detailed per-contour metrics")
):
    """
    Analyze segmentation masks and provide quality metrics and insights
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Prepare masks to analyze
        masks_to_analyze = []
        if index is not None:
            # Check specific mask
            if index < 0 or index >= len(session.masks):
                raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
            masks_to_analyze = [(index, session.masks[index])]
        else:
            # Analyze all masks
            masks_to_analyze = [(i, mask) for i, mask in enumerate(session.masks)]
        
        # Prepare analysis results
        analysis_results = []
        
        for mask_idx, mask in masks_to_analyze:
            # Get image dimensions
            height, width = mask.shape[:2]
            
            # Calculate basic statistics
            mask_area = np.sum(mask > 0)
            total_area = height * width
            coverage_percent = (mask_area / total_area) * 100
            
            # Get contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            contour_count = len(contours)
            outer_contours = 0
            inner_contours = 0
            total_perimeter = 0
            complexity_sum = 0
            max_contour_area = 0
            min_contour_area = float('inf') if contour_count > 0 else 0
            contour_details = []
            
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                
                for i, contour in enumerate(contours):
                    # Calculate contour metrics
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    is_inner = hierarchy[i][3] != -1  # Has parent
                    
                    # Update summary metrics
                    if is_inner:
                        inner_contours += 1
                    else:
                        outer_contours += 1
                    
                    total_perimeter += perimeter
                    max_contour_area = max(max_contour_area, area)
                    if area > 0:
                        min_contour_area = min(min_contour_area, area)
                    
                    # Circularity/complexity (1.0 is a perfect circle)
                    if area > 0:
                        complexity = perimeter**2 / (4 * np.pi * area)
                        complexity_sum += complexity
                    
                    # Add detailed contour info if requested
                    if detailed:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate convex hull for convexity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        convexity = area / hull_area if hull_area > 0 else 1.0
                        
                        contour_details.append({
                            "index": i,
                            "is_inner": is_inner,
                            "parent": hierarchy[i][3],
                            "area": float(area),
                            "perimeter": float(perimeter),
                            "complexity": float(perimeter**2 / (4 * np.pi * area)) if area > 0 else 0,
                            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "centroid": {
                                "x": int(np.mean(contour[:, 0, 0])),
                                "y": int(np.mean(contour[:, 0, 1]))
                            },
                            "convexity": float(convexity),
                            "point_count": len(contour)
                        })
            
            # Calculate average complexity
            avg_complexity = complexity_sum / contour_count if contour_count > 0 else 0
            
            # Prepare mask analysis
            mask_analysis = {
                "index": mask_idx,
                "dimensions": {"width": width, "height": height},
                "area": {
                    "mask_area": int(mask_area),
                    "total_area": int(total_area),
                    "coverage_percent": float(coverage_percent)
                },
                "contours": {
                    "total_count": contour_count,
                    "outer_count": outer_contours,
                    "inner_count": inner_contours,
                    "total_perimeter": float(total_perimeter),
                    "max_area": float(max_contour_area),
                    "min_area": float(min_contour_area),
                    "avg_complexity": float(avg_complexity)
                }
            }
            
            # Add detailed contour data if requested
            if detailed:
                mask_analysis["contour_details"] = contour_details
            
            analysis_results.append(mask_analysis)
        
        # Prepare response
        response_data = {
            "session_id": session_id,
            "analysis_count": len(analysis_results),
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-08 14:23:14",
            "masks": analysis_results
        }
        
        # Calculate summary statistics across all analyzed masks
        if len(analysis_results) > 1:
            total_contours = sum(m["contours"]["total_count"] for m in analysis_results)
            avg_contours_per_mask = total_contours / len(analysis_results)
            avg_coverage = sum(m["area"]["coverage_percent"] for m in analysis_results) / len(analysis_results)
            
            response_data["summary"] = {
                "total_masks": len(analysis_results),
                "total_contours": total_contours,
                "avg_contours_per_mask": float(avg_contours_per_mask),
                "avg_coverage_percent": float(avg_coverage)
            }
        
        return response_data
    except Exception as e:
        logger.error(f"Error analyzing segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


###############################################################################################
@app.get("/export_all/{session_id}")
async def export_all(
    session_id: str = FastAPIPath(..., description="Session ID"),
    format: str = Query("zip", description="Export format: zip, svg, or pdf")
):
    """
    Export all masks for a session in the requested format
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if there are masks to export
        if len(session.masks) == 0:
            raise HTTPException(status_code=400, detail="No masks found for this session")
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Export based on format
        timestamp = int(time.time())
        
        if format.lower() == "zip":
            # Create a temporary directory for export files
            temp_dir = export_dir / "temp_export"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Export each mask
                for i, mask in enumerate(session.masks):
                    # Save as PNG
                    png_path = temp_dir / f"mask_{i}.png"
                    cv2.imwrite(str(png_path), mask)
                    
                    # Create overlay version if original image exists
                    if session.original_image is not None:
                        overlay_image = session.original_image.copy()
                        overlay_color = [0, 0, 255]  # Red in BGR
                        
                        # Create colored overlay
                        color_mask = np.zeros_like(overlay_image)
                        color_mask[mask > 0] = overlay_color
                        
                        # Blend images
                        alpha = 0.5
                        overlay_image = cv2.addWeighted(overlay_image, 1, color_mask, alpha, 0)
                        
                        # Draw contours
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay_image, contours, -1, overlay_color, 2)
                        
                        # Save overlay
                        overlay_path = temp_dir / f"mask_{i}_overlay.png"
                        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
                
                # Create a ZIP file
                import zipfile
                zip_path = export_dir / f"all_masks_{timestamp}.zip"
                
                with zipfile.ZipFile(str(zip_path), "w") as zipf:
                    for file_path in temp_dir.glob("*"):
                        zipf.write(file_path, arcname=file_path.name)
                
                # Return the ZIP file
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=f"all_masks_export.zip"
                )
            finally:
                # Clean up temporary files
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        
        elif format.lower() == "svg":
            # Get image dimensions
            height, width = session.original_image.shape[:2] if session.original_image is not None else (600, 800)
            
            # Create SVG file with all masks
            svg_path = export_dir / f"all_masks_{timestamp}.svg"
            
            # Start SVG document
            svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description>
      <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">All Masks Export</dc:title>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">FETHl - API-SAM v2</dc:creator>
      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">2025-04-09 08:24:29</dc:date>
    </rdf:Description>
  </rdf:RDF>
</metadata>
"""
            
            # Include base64 encoded image if available
            if session.original_image is not None:
                import base64
                # Convert image to bytes
                success, buffer = cv2.imencode(".png", cv2.cvtColor(session.original_image, cv2.COLOR_RGB2BGR))
                if success:
                    img_bytes = buffer.tobytes()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    svg_content += f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}" opacity="0.5" />\n'
            
            # Add each mask as a group with its contours
            for i, mask in enumerate(session.masks):
                svg_content += f'<g id="mask_{i}" fill-opacity="0.3" stroke-width="2">\n'
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Generate random color (in HSL for better diversity)
                hue = (i * 30) % 360  # Space out the hues
                svg_color = f"hsl({hue}, 80%, 50%)"
                
                # Add each contour as a path
                for j, contour in enumerate(contours):
                    if len(contour) >= 3:  # Only add valid contours
                        # Convert contour to SVG path
                        path_data = "M "
                        for point in contour.reshape(-1, 2):
                            x, y = point
                            path_data += f"{x},{y} "
                        path_data += "Z"  # Close the path
                        
                        # Add filled and outlined path
                        svg_content += f'  <path d="{path_data}" fill="{svg_color}" stroke="{svg_color}" />\n'
                
                svg_content += '</g>\n'
            
            # Close SVG document
            svg_content += "</svg>"
            
            # Write SVG file
            with open(svg_path, "w") as f:
                f.write(svg_content)
            
            # Return SVG file
            return FileResponse(
                svg_path,
                media_type="image/svg+xml",
                filename=f"all_masks_export.svg"
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting all masks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/export_svg/{session_id}")
async def export_svg(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to export"),
    include_image: bool = Query(False, description="Include background image in export")
):
    """
    Export a mask as an SVG file
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Get the mask
        mask = session.masks[index]
        
        # Create SVG file
        timestamp = int(time.time())
        svg_path = export_dir / f"mask_{index}_export_{timestamp}.svg"
        
        # Create simple SVG content
        # Get image dimensions
        height, width = mask.shape[:2]
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Start SVG document
        svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description>
      <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Mask Export</dc:title>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">FETHl - API-SAM v2</dc:creator>
      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">2025-04-09 08:24:29</dc:date>
    </rdf:Description>
  </rdf:RDF>
</metadata>
"""
        
        # Include base64 encoded image if requested
        if include_image and session.original_image is not None:
            import base64
            # Convert the image to bytes
            success, buffer = cv2.imencode(".png", cv2.cvtColor(session.original_image, cv2.COLOR_RGB2BGR))
            if success:
                img_bytes = buffer.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                svg_content += f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}" />\n'
        
        # Add each contour as a path
        for i, contour in enumerate(contours):
            # Convert contour to SVG path
            path_data = "M "
            for point in contour.reshape(-1, 2):
                x, y = point
                path_data += f"{x},{y} "
            path_data += "Z"  # Close the path
            
            # Add path to SVG with random color
            import random
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            svg_content += f'<path d="{path_data}" fill="none" stroke="rgb({r},{g},{b})" stroke-width="2" />\n'
        
        # Close SVG document
        svg_content += "</svg>"
        
        # Write SVG file
        with open(svg_path, "w") as f:
            f.write(svg_content)
        
        # Return SVG file
        return FileResponse(
            svg_path,
            media_type="image/svg+xml",
            filename=f"mask_{index}_export.svg"
        )
    except Exception as e:
        # Log the error
        logger.error(f"Error exporting SVG: {str(e)}")
        # Return error
        raise HTTPException(status_code=500, detail=f"Error exporting SVG: {str(e)}")

@app.get("/export_png/{session_id}")
async def export_png(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to export"),
    overlay: bool = Query(True, description="Overlay mask on the original image")
):
    """
    Export a mask as a PNG image
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Get the mask
        mask = session.masks[index]
        
        # Create export image
        if overlay and session.original_image is not None:
            # Create a copy of the original image
            export_image = session.original_image.copy()
            
            # Create a colored overlay
            overlay_color = [0, 0, 255]  # Red in BGR
            color_mask = np.zeros_like(export_image)
            color_mask[mask > 0] = overlay_color
            
            # Blend the original image with the mask
            alpha = 0.5  # Transparency factor
            export_image = cv2.addWeighted(export_image, 1, color_mask, alpha, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(export_image, contours, -1, overlay_color, 2)
        else:
            # Just the mask
            export_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Save image
        timestamp = int(time.time())
        export_path = export_dir / f"mask_{index}_export_{timestamp}.png"
        cv2.imwrite(str(export_path), cv2.cvtColor(export_image, cv2.COLOR_RGB2BGR))
        
        # Return file
        return FileResponse(
            export_path,
            media_type="image/png",
            filename=f"mask_{index}_export.png"
        )
    except Exception as e:
        # Log the error
        logger.error(f"Error exporting PNG: {str(e)}")
        # Return error
        raise HTTPException(status_code=500, detail=f"Error exporting PNG: {str(e)}")

# Add this endpoint for mask editing functionality
@app.post("/editor/update_contour/{session_id}")
async def update_contour(
    session_id: str = FastAPIPath(..., description="Session ID"),
    mask_index: int = Query(..., description="Index of the mask to edit"),
    operation: str = Query(..., description="Edit operation: add_point, remove_point, move_point"),
    point_index: Optional[int] = Query(None, description="Index of the point for remove/move operations"),
    point_x: Optional[float] = Query(None, description="X coordinate for new/moved point"),
    point_y: Optional[float] = Query(None, description="Y coordinate for new/moved point"),
    contour_index: Optional[int] = Query(0, description="Index of the contour to edit")
):
    """
    Edit contours with proper coordinate handling
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if mask_index < 0 or mask_index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {mask_index} not found")
        
        # Get the mask
        mask = session.masks[mask_index]
        mask_path = session.mask_paths[mask_index]
        
        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if the specified contour exists
        if contour_index < 0 or contour_index >= len(contours):
            raise HTTPException(status_code=404, detail=f"Contour with index {contour_index} not found")
        
        # Get the contour to edit
        contour = contours[contour_index]
        
        # Perform the requested operation
        if operation == "add_point" and point_x is not None and point_y is not None:
            # Convert to integers (OpenCV requires integer coordinates)
            x, y = int(point_x), int(point_y)
            
            # Find where to insert the new point
            # Find the closest point on any segment of the contour
            min_dist = float('inf')
            insert_idx = 0
            
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]
                
                # Calculate distance from (x,y) to the line segment p1-p2
                # This is a simplified distance calculation
                dist = np.linalg.norm(np.cross(p2-p1, p1-[x,y]))/np.linalg.norm(p2-p1)
                
                if dist < min_dist:
                    min_dist = dist
                    insert_idx = (i + 1) % len(contour)
            
            # Insert the new point
            new_contour = np.insert(contour, insert_idx, [[x, y]], axis=0)
            contours[contour_index] = new_contour
            
            logger.info(f"Added point ({x},{y}) at index {insert_idx}")
            
        elif operation == "remove_point" and point_index is not None:
            # Check if the point index is valid
            if point_index < 0 or point_index >= len(contour):
                raise HTTPException(status_code=400, detail=f"Point index {point_index} out of range")
                
            # Remove the point (don't allow removing if it would result in less than 3 points)
            if len(contour) <= 3:
                raise HTTPException(status_code=400, detail="Cannot remove point from contour with only 3 points")
                
            # Delete the point
            new_contour = np.delete(contour, point_index, axis=0)
            contours[contour_index] = new_contour
            
            logger.info(f"Removed point at index {point_index}")
            
        elif operation == "move_point" and point_index is not None and point_x is not None and point_y is not None:
            # Check if the point index is valid
            if point_index < 0 or point_index >= len(contour):
                raise HTTPException(status_code=400, detail=f"Point index {point_index} out of range")
                
            # Convert to integers
            x, y = int(point_x), int(point_y)
            
            # Update the point
            contour[point_index] = [[x, y]]
            
            logger.info(f"Moved point at index {point_index} to ({x},{y})")
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
        
        # Create a new mask from the edited contours
        height, width = mask.shape
        new_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(new_mask, contours, -1, 255, -1)
        
        # Save the new mask
        cv2.imwrite(mask_path, new_mask)
        
        # Update the mask in session
        session.masks[mask_index] = new_mask
        
        # Return success with updated contour info
        return {
            "message": f"Successfully performed {operation} on contour {contour_index}",
            "contour_points": contours[contour_index].reshape(-1, 2).tolist(),
            "contour_count": len(contours),
            "point_count": len(contours[contour_index])
        }
    except Exception as e:
        logger.error(f"Error updating contour: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Contour edit error: {str(e)}")


@app.post("/editor/edit_contours/{session_id}")
async def edit_contours(
    session_id: str = FastAPIPath(..., description="Session ID"),
    mask_index: int = Query(..., description="Index of the mask to edit"),
    edit_data: Dict = Body(..., description="Contour edit data and operations")
):
    """
    Advanced contour editing with support for multiple operations
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if mask_index < 0 or mask_index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {mask_index} not found")
        
        # Get the mask
        mask = session.masks[mask_index]
        mask_path = session.mask_paths[mask_index]
        
        # Extract operations from edit_data
        operations = edit_data.get("operations", [])
        contour_names = edit_data.get("names", {})
        contour_groups = edit_data.get("groups", {})
        
        # Find contours in the mask with hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] if hierarchy is not None else None
        
        # Process each operation
        for op in operations:
            op_type = op.get("type")
            contour_idx = op.get("contour_index", 0)
            
            # Check if contour exists
            if contour_idx < 0 or contour_idx >= len(contours):
                continue  # Skip invalid operations
            
            # Get the contour
            contour = contours[contour_idx]
            
            if op_type == "add_point":
                # Add a point
                x, y = int(op.get("x")), int(op.get("y"))
                insert_idx = op.get("insert_index", len(contour))
                
                # Ensure valid insert index
                insert_idx = min(max(0, insert_idx), len(contour))
                
                # Insert the point
                contour = np.insert(contour, insert_idx, [[x, y]], axis=0)
                contours[contour_idx] = contour
            
            elif op_type == "remove_point":
                # Remove a point
                point_idx = op.get("point_index")
                
                # Ensure we keep at least 3 points
                if len(contour) > 3 and 0 <= point_idx < len(contour):
                    contour = np.delete(contour, point_idx, axis=0)
                    contours[contour_idx] = contour
            
            elif op_type == "move_point":
                # Move a point
                point_idx = op.get("point_index")
                x, y = int(op.get("x")), int(op.get("y"))
                
                if 0 <= point_idx < len(contour):
                    contour[point_idx] = [[x, y]]
            
            elif op_type == "smooth_contour":
                # Smooth the contour
                smoothing_factor = op.get("factor", 0.2)
                
                # Simple smoothing by averaging neighboring points
                if len(contour) > 3:
                    smoothed = np.copy(contour)
                    for i in range(len(contour)):
                        prev = (i - 1) % len(contour)
                        next_idx = (i + 1) % len(contour)
                        
                        # Weighted average
                        curr_pt = contour[i][0] * (1 - smoothing_factor)
                        neighbor_avg = (contour[prev][0] + contour[next_idx][0]) * smoothing_factor / 2
                        
                        smoothed[i][0] = curr_pt + neighbor_avg
                    
                    contours[contour_idx] = smoothed
            
            elif op_type == "rename_contour":
                # Just store the name (processed later)
                pass
        
        # Create a new mask with updated contours
        height, width = mask.shape
        new_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(new_mask, contours, -1, 255, -1)
        
        # Save the new mask
        cv2.imwrite(str(mask_path), new_mask)
        
        # Update the mask in session
        session.masks[mask_index] = new_mask
        
        # Store contour metadata if provided
        if hasattr(session, 'contour_metadata'):
            metadata = session.contour_metadata
        else:
            metadata = {}
            session.contour_metadata = metadata
        
        # Initialize or update mask-specific metadata
        if str(mask_index) not in metadata:
            metadata[str(mask_index)] = {}
        
        mask_metadata = metadata[str(mask_index)]
        
        # Update contour names
        if contour_names:
            if 'names' not in mask_metadata:
                mask_metadata['names'] = {}
            
            for idx, name in contour_names.items():
                mask_metadata['names'][idx] = name
        
        # Update contour groups
        if contour_groups:
            if 'groups' not in mask_metadata:
                mask_metadata['groups'] = {}
            
            for group_name, contour_indices in contour_groups.items():
                mask_metadata['groups'][group_name] = contour_indices
        
        # Prepare response with updated contour data
        contour_data = []
        for i, contour in enumerate(contours):
            points = contour.reshape(-1, 2).tolist()
            
            # Get metadata for this contour
            name = mask_metadata.get('names', {}).get(str(i), f"Contour {i}")
            
            # Determine parent-child relationships
            parent = -1
            if hierarchy is not None and i < len(hierarchy):
                parent = int(hierarchy[i][3])
            
            # Find which groups this contour belongs to
            groups = []
            for group_name, members in mask_metadata.get('groups', {}).items():
                if str(i) in members:
                    groups.append(group_name)
            
            contour_data.append({
                "index": i,
                "points": points,
                "name": name,
                "parent": parent,
                "groups": groups,
                "point_count": len(points)
            })
        
        return {
            "message": "Contours updated successfully",
            "mask_index": mask_index,
            "contour_count": len(contours),
            "contours": contour_data,
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-09 09:17:50"
        }
    except Exception as e:
        logger.error(f"Error editing contours: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Contour editing error: {str(e)}")

@app.get("/visualization/all_masks/{session_id}")
async def visualize_all_masks(
    session_id: str = FastAPIPath(..., description="Session ID"),
    overlay: bool = Query(True, description="Overlay masks on original image"),
    use_colors: bool = Query(True, description="Use different colors for each mask"),
    label_contours: bool = Query(True, description="Label contours with names/indices"),
    include_metadata: bool = Query(True, description="Include mask and contour metadata")
):
    """
    Create a visualization of all masks in a session with customizable options
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if there are masks to visualize
        if len(session.masks) == 0:
            raise HTTPException(status_code=400, detail="No masks found for this session")
        
        # Create visualization directories
        vis_dir = Path(BASE_IMAGE_PATH) / session_id / "visualization"
        vis_dir.mkdir(exist_ok=True)
        
        # Prepare the visualization canvas
        if overlay and session.original_image is not None:
            # Use original image as background
            vis_image = session.original_image.copy()
        else:
            # Use white canvas
            if session.original_image is not None:
                height, width = session.original_image.shape[:2]
            else:
                # Use dimensions of first mask
                height, width = session.masks[0].shape[:2]
            
            vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Generate color palette for masks
        import random
        colors = []
        for i in range(len(session.masks)):
            if use_colors:
                # Generate distinct colors using HSV color space
                hue = (i * 30) % 180
                sat = 200 + (i % 3) * 20
                val = 200 + (i % 2) * 55
                
                # Convert to BGR
                color_hsv = np.uint8([[[hue, sat, val]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            else:
                # Default to blue
                color_bgr = [255, 0, 0]  # Blue in BGR
            
            colors.append(color_bgr)
        
        # Draw each mask with its contours
        for i, mask in enumerate(session.masks):
            # Get color for this mask
            color = colors[i]
            
            # Create mask overlay with reduced opacity
            if overlay:
                # Create colored overlay
                overlay = np.zeros_like(vis_image)
                overlay[mask > 0] = color
                
                # Apply overlay with transparency
                alpha = 0.3
                vis_image = cv2.addWeighted(vis_image, 1, overlay, alpha, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
            
            # Label contours if requested
            if label_contours:
                # Get metadata for this mask
                metadata = getattr(session, 'contour_metadata', {}).get(str(i), {})
                names = metadata.get('names', {})
                
                for j, contour in enumerate(contours):
                    # Find centroid for text placement
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Get name or default to index
                        name = names.get(str(j), f"M{i}C{j}")
                        
                        # Draw text with contrasting background
                        cv2.putText(vis_image, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, [255, 255, 255], 2)  # White outline
                        cv2.putText(vis_image, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, [0, 0, 0], 1)  # Black text
        
        # Generate timestamp for unique filename
        timestamp = int(time.time())
        vis_path = vis_dir / f"all_masks_visualization_{timestamp}.png"
        
        # Save visualization
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Prepare metadata response
        response = {
            "message": "All masks visualization created successfully",
            "visualization_path": str(vis_path),
            "mask_count": len(session.masks),
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-09 09:17:50"
        }
        
        # Include metadata if requested
        if include_metadata:
            masks_metadata = []
            for i, mask in enumerate(session.masks):
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                mask_meta = {
                    "index": i,
                    "contour_count": len(contours),
                    "area": int(np.sum(mask > 0)),
                    "color": colors[i]
                }
                
                # Add contour metadata
                if hasattr(session, 'contour_metadata'):
                    mask_meta.update(session.contour_metadata.get(str(i), {}))
                
                masks_metadata.append(mask_meta)
            
            response["masks_metadata"] = masks_metadata
        
        # Return visualization path and metadata
        return response
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@app.get("/visualization/mask_detail/{session_id}")
async def mask_detail(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to view"),
    show_points: bool = Query(True, description="Show contour points"),
    show_hierarchy: bool = Query(True, description="Show contour hierarchy"),
    highlight_inner: bool = Query(True, description="Highlight inner contours")
):
    """
    Create a detailed visualization of a specific mask with points and hierarchy
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask
        mask = session.masks[index]
        
        # Create visualization directories
        vis_dir = Path(BASE_IMAGE_PATH) / session_id / "visualization"
        vis_dir.mkdir(exist_ok=True)
        
        # Prepare the visualization canvas
        if session.original_image is not None:
            # Use original image as background with reduced opacity
            vis_image = session.original_image.copy()
            
            # Apply semi-transparent white overlay to dim the background
            white_overlay = np.ones_like(vis_image) * 255
            alpha = 0.7
            vis_image = cv2.addWeighted(vis_image, 1-alpha, white_overlay, alpha, 0)
        else:
            # Use white canvas
            height, width = mask.shape[:2]
            vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get metadata for this mask
        metadata = getattr(session, 'contour_metadata', {}).get(str(index), {})
        contour_names = metadata.get('names', {})
        
        # Process contours and their hierarchy
        if contours:
            # Draw contours with different colors based on hierarchy
            if hierarchy is not None and show_hierarchy:
                hierarchy = hierarchy[0]  # Flatten hierarchy
                
                # Draw contours based on hierarchy
                for i, contour in enumerate(contours):
                    # Determine hierarchy level (depth)
                    depth = 0
                    parent = hierarchy[i][3]
                    while parent != -1 and depth < 10:  # Prevent infinite loops
                        depth += 1
                        parent = hierarchy[parent][3]
                    
                    # Choose color based on depth
                    if highlight_inner and depth > 0:
                        # Inner contours in red
                        color = [0, 0, 255]  # Red in BGR
                        thickness = 2
                    else:
                        # Outer contours in blue
                        color = [255, 0, 0]  # Blue in BGR
                        thickness = 2
                    
                    # Draw contour
                    cv2.drawContours(vis_image, contours, i, color, thickness)
                    
                    # Draw points if requested
                    if show_points:
                        for j, point in enumerate(contour):
                            x, y = point[0]
                            # Draw point
                            cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)  # Green point
                            
                            # Label every nth point to avoid clutter
                            if j % 5 == 0:
                                cv2.putText(vis_image, str(j), (x+5, y+5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Label contour
                    name = contour_names.get(str(i), f"C{i} (D{depth})")
                    
                    # Find a good position for the label
                    if len(contour) > 0:
                        # Use center of bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        cx, cy = x + w//2, y + h//2
                        
                        # Draw label with background
                        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(vis_image, (cx-2, cy-text_size[1]-2), 
                                     (cx+text_size[0]+2, cy+2), (255, 255, 255), -1)
                        cv2.putText(vis_image, name, (cx, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # Draw all contours in blue
                cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 2)
                
                # Draw points if requested
                if show_points:
                    for i, contour in enumerate(contours):
                        for j, point in enumerate(contour):
                            x, y = point[0]
                            cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
                            
                            # Label every nth point
                            if j % 5 == 0:
                                cv2.putText(vis_image, str(j), (x+5, y+5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Generate timestamp for unique filename
        timestamp = int(time.time())
        vis_path = vis_dir / f"mask_{index}_detail_{timestamp}.png"
        
        # Save visualization
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Return visualization path and metadata
        return {
            "message": "Mask detail visualization created",
            "visualization_path": str(vis_path),
            "mask_index": index,
            "contour_count": len(contours) if contours else 0,
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-09 09:17:50"
        }
    except Exception as e:
        logger.error(f"Error creating mask detail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mask detail error: {str(e)}")
    
@app.post("/editor/contour_groups/{session_id}")
async def manage_contour_groups(
    session_id: str = FastAPIPath(..., description="Session ID"),
    mask_index: int = Query(..., description="Index of the mask to manage groups for"),
    operation: str = Body(..., description="Group operation: create, add, remove, delete"),
    group_name: str = Body(..., description="Name of the group to manage"),
    contour_indices: List[int] = Body([], description="Indices of contours to include in group")
):
    """
    Manage coherent groups of contours for a mask
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if mask_index < 0 or mask_index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {mask_index} not found")
        
        # Initialize contour metadata if not present
        if not hasattr(session, 'contour_metadata'):
            session.contour_metadata = {}
        
        if str(mask_index) not in session.contour_metadata:
            session.contour_metadata[str(mask_index)] = {}
        
        mask_metadata = session.contour_metadata[str(mask_index)]
        
        # Initialize groups if not present
        if 'groups' not in mask_metadata:
            mask_metadata['groups'] = {}
        
        groups = mask_metadata['groups']
        
        # Perform the requested operation
        if operation == "create":
            # Create a new group
            if group_name in groups:
                raise HTTPException(status_code=400, detail=f"Group '{group_name}' already exists")
            
            groups[group_name] = [str(idx) for idx in contour_indices]
            message = f"Created group '{group_name}' with {len(contour_indices)} contours"
        
        elif operation == "add":
            # Add contours to existing group
            if group_name not in groups:
                groups[group_name] = []
            
            # Convert to string and ensure uniqueness
            existing = set(groups[group_name])
            new_indices = [str(idx) for idx in contour_indices if str(idx) not in existing]
            
            groups[group_name].extend(new_indices)
            message = f"Added {len(new_indices)} contours to group '{group_name}'"
        
        elif operation == "remove":
            # Remove contours from a group
            if group_name not in groups:
                raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")
            
            # Convert to strings for comparison
            to_remove = set(str(idx) for idx in contour_indices)
            original_count = len(groups[group_name])
            groups[group_name] = [idx for idx in groups[group_name] if idx not in to_remove]
            
            removed_count = original_count - len(groups[group_name])
            message = f"Removed {removed_count} contours from group '{group_name}'"
        
        elif operation == "delete":
            # Delete an entire group
            if group_name not in groups:
                raise HTTPException(status_code=404, detail=f"Group '{group_name}' not found")
            
            del groups[group_name]
            message = f"Deleted group '{group_name}'"
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
        
        # Return updated groups data
        return {
            "message": message,
            "mask_index": mask_index,
            "groups": groups,
            "timestamp": time.time(),
            "user": "FETHl",
            "date": "2025-04-09 09:17:50"
        }
    except Exception as e:
        logger.error(f"Error managing contour groups: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Group management error: {str(e)}")
    
@app.get("/export/multiple_svg/{session_id}")
async def export_multiple_svg(
    session_id: str = FastAPIPath(..., description="Session ID"),
    mask_indices: str = Query(..., description="Comma-separated indices of masks to export"),
    include_image: bool = Query(False, description="Include background image in export"),
    include_hierarchy: bool = Query(True, description="Include contour hierarchy information"),
    include_groups: bool = Query(True, description="Use group information for contours"),
    use_layers: bool = Query(True, description="Organize contours in SVG layers")
):
    """
    Export multiple masks as a single SVG file with proper organization
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Parse mask indices
        try:
            indices = [int(idx.strip()) for idx in mask_indices.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid mask indices format. Use comma-separated numbers.")
        
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(session.masks):
                raise HTTPException(status_code=404, detail=f"Mask with index {idx} not found")
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = int(time.time())
        svg_path = export_dir / f"masks_export_{timestamp}.svg"
        
        # Determine SVG dimensions from image or first mask
        if session.original_image is not None:
            height, width = session.original_image.shape[:2]
        else:
            height, width = session.masks[indices[0]].shape[:2]
        
        # Start SVG document
        svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description>
      <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Multiple Mask Export</dc:title>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">FETHl - API-SAM v2</dc:creator>
      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">2025-04-09 09:17:50</dc:date>
    </rdf:Description>
  </rdf:RDF>
</metadata>
"""
        
        # Include original image if requested
        if include_image and session.original_image is not None:
            import base64
            # Convert image to bytes
            success, buffer = cv2.imencode(".png", cv2.cvtColor(session.original_image, cv2.COLOR_RGB2BGR))
            if success:
                img_bytes = buffer.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                svg_content += f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}" opacity="0.3" />\n'
        
        # Add SVG definitions for markers (optional)
        svg_content += """<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#000" />
  </marker>
</defs>
"""
        
        # Process each requested mask
        for mask_idx in indices:
            mask = session.masks[mask_idx]
            
            # Get contours with hierarchy
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = hierarchy[0] if hierarchy is not None else None
            
            # Get metadata for this mask
            mask_metadata = getattr(session, 'contour_metadata', {}).get(str(mask_idx), {})
            contour_names = mask_metadata.get('names', {})
            contour_groups = mask_metadata.get('groups', {})
            
            # Reverse lookup for groups (which group each contour belongs to)
            contour_to_groups = {}
            for group_name, members in contour_groups.items():
                for member in members:
                    if member not in contour_to_groups:
                        contour_to_groups[member] = []
                    contour_to_groups[member].append(group_name)
            
            # Start a mask group in SVG
            svg_content += f'<g id="mask_{mask_idx}" class="mask">\n'
            
            # Organize contours by hierarchy and groups
            if use_layers:
                # First layer: groups of contours
                if include_groups and contour_groups:
                    svg_content += f'  <g id="mask_{mask_idx}_groups" class="contour_groups">\n'
                    
                    # Process each group
                    for group_name, members in contour_groups.items():
                        # Start group
                        svg_content += f'    <g id="group_{group_name}" class="contour_group">\n'
                        
                        # Generate a consistent color for this group (using hash of name)
                        import hashlib
                        name_hash = int(hashlib.md5(group_name.encode()).hexdigest(), 16)
                        hue = name_hash % 360
                        group_color = f"hsl({hue}, 70%, 50%)"
                        
                        # Add title for the group
                        svg_content += f'      <title>{group_name}</title>\n'
                        
                        # Add contours in this group
                        for member in members:
                            try:
                                i = int(member)
                                if i >= 0 and i < len(contours):
                                    contour = contours[i]
                                    # Convert contour to SVG path
                                    path_data = "M "
                                    for point in contour.reshape(-1, 2):
                                        x, y = point
                                        path_data += f"{x},{y} "
                                    path_data += "Z"  # Close the path
                                    
                                    # Get contour name
                                    name = contour_names.get(member, f"Contour {member}")
                                    
                                    # Add path to SVG
                                    svg_content += f'      <path d="{path_data}" fill="none" stroke="{group_color}" stroke-width="2">\n'
                                    svg_content += f'        <title>{name} (Group: {group_name})</title>\n'
                                    svg_content += f'      </path>\n'
                            except (ValueError, IndexError):
                                # Skip invalid indices
                                pass
                        
                        # End group
                        svg_content += f'    </g>\n'
                    
                    # End groups layer
                    svg_content += f'  </g>\n'
                
                # Second layer: contours by hierarchy
                if include_hierarchy and hierarchy is not None:
                    svg_content += f'  <g id="mask_{mask_idx}_hierarchy" class="contour_hierarchy">\n'
                    
                    # Find root contours (no parent)
                    def process_contour_tree(idx, level=0):
                        nonlocal svg_content
                        
                        if idx < 0 or idx >= len(contours):
                            return
                        
                        contour = contours[idx]
                        
                        # Convert contour to SVG path
                        path_data = "M "
                        for point in contour.reshape(-1, 2):
                            x, y = point
                            path_data += f"{x},{y} "
                        path_data += "Z"  # Close the path
                        
                        # Get contour name and groups
                        name = contour_names.get(str(idx), f"Contour {idx}")
                        groups = contour_to_groups.get(str(idx), [])
                        
                        # Generate color based on hierarchy level
                        level_color = f"hsl({level * 30 % 360}, 80%, {max(30, 70 - level * 10)}%)"
                        
                        # Start a group for this contour and its children
                        svg_content += f'    <g id="contour_{idx}" class="hierarchy_level_{level}">\n'
                        
                        # Add path for this contour
                        svg_content += f'      <path d="{path_data}" fill="none" stroke="{level_color}" stroke-width="{max(0.5, 2.0 - level * 0.3)}">\n'
                        
                        # Add title with metadata
                        svg_content += f'        <title>{name} (Level: {level}, Groups: {", ".join(groups)})</title>\n'
                        svg_content += f'      </path>\n'
                        
                        # Process children
                        for i, h in enumerate(hierarchy):
                            if h[3] == idx:  # if parent is current contour
                                process_contour_tree(i, level + 1)
                        
                        # End contour group
                        svg_content += f'    </g>\n'
                    
                    # Process all root contours
                    for i, h in enumerate(hierarchy):
                        if h[3] == -1:  # No parent
                            process_contour_tree(i)
                    
                    # End hierarchy layer
                    svg_content += f'  </g>\n'
            else:
                # Simple flat structure
                for i, contour in enumerate(contours):
                    # Convert contour to SVG path
                    path_data = "M "
                    for point in contour.reshape(-1, 2):
                        x, y = point
                        path_data += f"{x},{y} "
                    path_data += "Z"  # Close the path
                    
                    # Generate color
                    hue = (mask_idx * 40 + i * 20) % 360
                    color = f"hsl({hue}, 70%, 50%)"
                    
                    # Get name
                    name = contour_names.get(str(i), f"Contour {i}")
                    
                    # Add path
                    svg_content += f'  <path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.5">\n'
                    svg_content += f'    <title>{name}</title>\n'
                    svg_content += f'  </path>\n'
            
            # End mask group
            svg_content += f'</g>\n'
        
        # End SVG document
        svg_content += "</svg>"
        
        # Write SVG file
        with open(svg_path, "w") as f:
            f.write(svg_content)
        
        # Return the SVG file
        return FileResponse(
            svg_path,
            media_type="image/svg+xml",
            filename=f"masks_export_{timestamp}.svg"
        )
    except Exception as e:
        logger.error(f"Error exporting SVG: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"SVG export error: {str(e)}")
@app.get("/export/formats/{session_id}")
async def export_formats(
    session_id: str = FastAPIPath(..., description="Session ID"),
    mask_indices: str = Query(..., description="Comma-separated indices of masks to export"),
    format: str = Query(..., pattern="^(png|svg|dxf|pdf|ai)$", description="Export format"),
    include_image: bool = Query(False, description="Include background image in export"),
    include_hierarchy: bool = Query(True, description="Preserve contour hierarchy"),
    scale: float = Query(1.0, description="Scale factor for export")
):
    """
    Export masks in various professional formats with extended options
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Parse mask indices
        try:
            indices = [int(idx.strip()) for idx in mask_indices.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid mask indices format. Use comma-separated numbers.")
        
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(session.masks):
                raise HTTPException(status_code=404, detail=f"Mask with index {idx} not found")
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = int(time.time())
        
        # Export based on format
        if format.lower() == "png":
            # Create a composite PNG image with all selected masks
            
            # Determine canvas size
            if session.original_image is not None:
                height, width = session.original_image.shape[:2]
                # Apply scaling if needed
                if scale != 1.0:
                    width, height = int(width * scale), int(height * scale)
            else:
                # Use dimensions of first mask
                height, width = session.masks[indices[0]].shape[:2]
                if scale != 1.0:
                    width, height = int(width * scale), int(height * scale)
            
            # Create canvas
            if include_image and session.original_image is not None:
                # Use original image as background
                original = session.original_image.copy()
                if scale != 1.0:
                    original = cv2.resize(original, (width, height))
                canvas = original
            else:
                # Use white canvas
                canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add each mask with different color
            for i, mask_idx in enumerate(indices):
                mask = session.masks[mask_idx]
                
                # Apply scaling if needed
                if scale != 1.0:
                    mask = cv2.resize(mask, (width, height))
                
                # Generate color using HSV for better distribution
                hue = (i * 30) % 180
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Create a colored overlay
                overlay = np.zeros_like(canvas)
                overlay[mask > 0] = color
                
                # Blend with transparency
                alpha = 0.4
                canvas = cv2.addWeighted(canvas, 1, overlay, alpha, 0)
                
                # Draw contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2)
            
            # Save PNG file
            png_path = export_dir / f"masks_export_{timestamp}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            
            # Return the PNG file
            return FileResponse(
                png_path,
                media_type="image/png",
                filename=f"masks_export_{timestamp}.png"
            )
        
        elif format.lower() == "svg":
            # This is already handled by export_multiple_svg endpoint
            # Redirect to that endpoint
            return RedirectResponse(url=f"/export/multiple_svg/{session_id}?mask_indices={mask_indices}&include_image={include_image}&include_hierarchy={include_hierarchy}&use_layers=true")
        
        elif format.lower() == "dxf":
            # Export as DXF for CAD systems
            try:
                # Try to use the ezdxf library if available
                import ezdxf
                
                # Create a new DXF document
                doc = ezdxf.new("R2010")
                msp = doc.modelspace()
                
                # Add metadata
                doc.header['$ACADVER'] = "AC1024"
                
                # Create layers for organization
                for mask_idx in indices:
                    # Create a layer for each mask
                    layer_name = f"MASK_{mask_idx}"
                    doc.layers.add(layer_name)
                    
                    # Set layer color
                    layer = doc.layers.get(layer_name)
                    layer.color = (mask_idx % 7) + 1  # ACI colors 1-7
                    
                    # Get the mask and its contours
                    mask = session.masks[mask_idx]
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process contours
                    for i, contour in enumerate(contours):
                        # Convert points to DXF format (scale if needed)
                        points = [(point[0][0] * scale, point[0][1] * scale) for point in contour]
                        
                        # Create polyline
                        polyline = msp.add_lwpolyline(points, dxfattribs={
                            "layer": layer_name,
                            "closed": True
                        })
                
                # Save the DXF file
                dxf_path = export_dir / f"masks_export_{timestamp}.dxf"
                doc.saveas(str(dxf_path))
                
                # Return the DXF file
                return FileResponse(
                    dxf_path,
                    media_type="application/dxf",
                    filename=f"masks_export_{timestamp}.dxf"
                )
            except ImportError:
                # Fallback to simpler DXF creation if ezdxf is not available
                raise HTTPException(status_code=501, detail="DXF export requires ezdxf library")
        
        elif format.lower() == "pdf":
            # Export as PDF
            try:
                # Try to use the reportlab library if available
                from reportlab.pdfgen import canvas as pdf_canvas
                from reportlab.lib.pagesizes import letter
                from reportlab.lib import colors
                
                # Create PDF file
                pdf_path = export_dir / f"masks_export_{timestamp}.pdf"
                c = pdf_canvas.Canvas(str(pdf_path), pagesize=letter)
                
                # Set metadata
                c.setTitle("SAM Segmentation Export")
                c.setAuthor("FETHl - API-SAM v2")
                c.setSubject("Mask Export")
                
                # Get page dimensions
                width, height = letter
                
                # Determine scale factor for fitting masks to page
                if session.original_image is not None:
                    img_height, img_width = session.original_image.shape[:2]
                else:
                    img_height, img_width = session.masks[indices[0]].shape[:2]
                
                # Calculate scale to fit on page with margins
                page_scale = min((width - 72) / img_width, (height - 72) / img_height) * scale
                
                # Add masks
                for i, mask_idx in enumerate(indices):
                    # Create a new page for each mask
                    if i > 0:
                        c.showPage()
                    
                    # Draw title
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(36, height - 36, f"Mask {mask_idx}")
                    
                    # Add timestamp
                    c.setFont("Helvetica", 10)
                    c.drawString(36, height - 54, f"Export Date: 2025-04-09 09:17:50")
                    
                    # Include original image if requested
                    if include_image and session.original_image is not None:
                        # Convert image to bytes
                        success, buffer = cv2.imencode(".png", cv2.cvtColor(session.original_image, cv2.COLOR_RGB2BGR))
                        if success:
                            from io import BytesIO
                            img_bytes = BytesIO(buffer.tobytes())
                            
                            # Draw image
                            img_x, img_y = 36, height - img_height * page_scale - 72
                            c.drawImage(img_bytes, img_x, img_y, 
                                        width=img_width * page_scale, 
                                        height=img_height * page_scale)
                    
                    # Get mask contours
                    mask = session.masks[mask_idx]
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours
                    for j, contour in enumerate(contours):
                        # Convert contour to path
                        path = c.beginPath()
                        
                        # Start path at first point
                        first_point = contour[0][0]
                        path.moveTo(36 + first_point[0] * page_scale, 
                                   height - 72 - first_point[1] * page_scale)
                        
                        # Add remaining points
                        for point in contour[1:]:
                            x, y = point[0]
                            path.lineTo(36 + x * page_scale, height - 72 - y * page_scale)
                        
                        # Close path
                        path.close()
                        
                        # Draw path
                        c.setStrokeColorRGB(0, 0, 1)  # Blue
                        c.setLineWidth(1)
                        c.drawPath(path)
                
                # Save the PDF
                c.save()
                
                # Return the PDF file
                return FileResponse(
                    pdf_path,
                    media_type="application/pdf",
                    filename=f"masks_export_{timestamp}.pdf"
                )
            except ImportError:
                raise HTTPException(status_code=501, detail="PDF export requires reportlab library")
        
        elif format.lower() == "ai":
            # Export as Adobe Illustrator (AI) format
            # AI format is essentially a specialized PDF, so we'll create a PDF with AI compatibility
            try:
                # Try to use the reportlab library if available
                from reportlab.pdfgen import canvas as pdf_canvas
                from reportlab.lib.pagesizes import letter
                
                # Create AI file (PDF with special header)
                ai_path = export_dir / f"masks_export_{timestamp}.ai"
                
                # Create PDF with AI compatibility
                c = pdf_canvas.Canvas(str(ai_path), pagesize=letter)
                
                # Add AI compatibility header
                c._doc.setProducer("Adobe Illustrator")
                c._doc.setCreator("Adobe Illustrator")
                
                # Set metadata
                c.setTitle("SAM Segmentation Export")
                c.setAuthor("FETHl - API-SAM v2")
                c.setSubject("Mask Export")
                
                # Get page dimensions
                width, height = letter
                
                # Determine scale factor for fitting masks to page
                if session.original_image is not None:
                    img_height, img_width = session.original_image.shape[:2]
                else:
                    img_height, img_width = session.masks[indices[0]].shape[:2]
                
                # Calculate scale to fit on page with margins
                page_scale = min((width - 72) / img_width, (height - 72) / img_height) * scale
                
                # Add masks
                for i, mask_idx in enumerate(indices):
                    # Create a new page for each mask
                    if i > 0:
                        c.showPage()
                    
                    # Draw title
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(36, height - 36, f"Mask {mask_idx}")
                    
                    # Get mask contours
                    mask = session.masks[mask_idx]
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours
                    for j, contour in enumerate(contours):
                        # Convert contour to path
                        path = c.beginPath()
                        
                        # Start path at first point
                        first_point = contour[0][0]
                        path.moveTo(36 + first_point[0] * page_scale, 
                                   height - 72 - first_point[1] * page_scale)
                        
                        # Add remaining points
                        for point in contour[1:]:
                            x, y = point[0]
                            path.lineTo(36 + x * page_scale, height - 72 - y * page_scale)
                        
                        # Close path
                        path.close()
                        
                        # Draw path
                        c.setStrokeColorRGB(0, 0, 1)  # Blue
                        c.setLineWidth(1)
                        c.drawPath(path)
                
                # Save the AI file
                c.save()
                
                # Return the AI file
                return FileResponse(
                    ai_path,
                    media_type="application/illustrator",
                    filename=f"masks_export_{timestamp}.ai"
                )
            except ImportError:
                raise HTTPException(status_code=501, detail="AI export requires reportlab library")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting in {format} format: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")
    
# Professional export to CAD-compatible formats
@app.get("/auto/export/cad/{session_id}")
async def export_to_cad(
    session_id: str = FastAPIPath(..., description="Session ID"),
    index: int = Query(..., description="Index of the mask to export"),
    format: str = Query("dxf", pattern="^(dxf|svg|step)$", description="CAD export format"),
    units: str = Query("mm", pattern="^(mm|cm|inch)$", description="Physical units"),
    scale: float = Query(1.0, description="Scale factor for export"),
    include_inner_contours: bool = Query(True, description="Include inner contours"),
    simplify: bool = Query(True, description="Simplify contours for CAD compatibility"),
    tolerance: float = Query(0.5, description="Tolerance for simplification"),
    include_metadata: bool = Query(True, description="Include metadata in export")
):
    """
    Export mask contours to CAD-compatible formats with professional options
    """
    try:
        # Check if the session exists
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        session.update_activity()
        
        # Check if the mask exists
        if index < 0 or index >= len(session.masks):
            raise HTTPException(status_code=404, detail=f"Mask with index {index} not found")
        
        # Get the mask
        mask = session.masks[index]
        
        # Extract contours
        contour_mode = cv2.RETR_TREE if include_inner_contours else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create export directory
        export_dir = Path(BASE_IMAGE_PATH) / session_id / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Prepare filename with timestamp
        timestamp = int(time.time())
        filename = f"cad_export_{index}_{timestamp}.{format}"
        export_path = export_dir / filename
        
        # Get image dimensions
        height, width = mask.shape[:2]
        
        # Prepare metadata
        metadata = {
            "title": f"SAM Segmentation Export - Mask {index}",
            "author": "FETHl - API-SAM v2",
            "date": "2025-04-08 14:23:14",
            "units": units,
            "scale": scale,
            "original_dimensions": {"width": width, "height": height}
        }
        
        # Process contours - simplify and organize
        processed_contours = []
        
        if contours:
            # Convert contours hierarchy to a tree if needed
            if include_inner_contours and hierarchy is not None:
                hierarchy = hierarchy[0]
                
                # Process each contour
                for i, contour in enumerate(contours):
                    # Simplify contour if requested
                    if simplify:
                        epsilon = tolerance * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Apply scale
                    if scale != 1.0:
                        contour = contour * scale
                    
                    # Store with hierarchy info
                    is_inner = hierarchy[i][3] != -1  # Has parent
                    parent_idx = hierarchy[i][3]
                    
                    processed_contours.append({
                        "points": contour,
                        "is_inner": is_inner,
                        "parent_idx": parent_idx
                    })
            else:
                # Just process external contours
                for contour in contours:
                    # Simplify contour if requested
                    if simplify:
                        epsilon = tolerance * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Apply scale
                    if scale != 1.0:
                        contour = contour * scale
                    
                    processed_contours.append({
                        "points": contour,
                        "is_inner": False,
                        "parent_idx": -1
                    })
        
        # Export based on format
        if format.lower() == "dxf":
            try:
                # Import DXF exporter
                from dxf_exporter import CADExporter
                
                # Configure exporter
                exporter = CADExporter(
                    units=units,
                    scale=scale,
                    line_thickness=0.25,
                    include_metadata=include_metadata
                )
                
                # Export to DXF
                exporter.export_to_dxf(processed_contours, str(export_path), metadata)
                
                return FileResponse(
                    export_path,
                    media_type="application/dxf",
                    filename=filename
                )
            except ImportError:
                # Fallback if dxf_exporter is not available
                logger.warning("DXF exporter module not available, using fallback method")
                
                # Basic DXF export using ezdxf if available
                try:
                    import ezdxf
                    
                    # Create DXF document
                    doc = ezdxf.new("R2010")
                    msp = doc.modelspace()
                    
                    # Add metadata
                    if include_metadata:
                        doc.header['$ACADVER'] = "AC1024"
                        doc.header['$INSUNITS'] = 4 if units == "mm" else 5 if units == "cm" else 1  # mm=4, cm=5, inch=1
                        
                        # Create text with metadata
                        msp.add_text(
                            f"SAM Export - {metadata['date']} - User: {metadata['author']}",
                            dxfattribs={"layer": "METADATA", "height": 5.0}
                        ).set_pos((0, -20))
                    
                    # Add contours as polylines
                    for idx, contour_data in enumerate(processed_contours):
                        contour = contour_data["points"]
                        is_inner = contour_data["is_inner"]
                        
                        # Convert contour to points
                        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
                        
                        # Create polyline
                        polyline = msp.add_lwpolyline(points, dxfattribs={
                            "layer": "INNER_CONTOURS" if is_inner else "OUTER_CONTOURS",
                            "color": 3 if is_inner else 5,  # 3=green, 5=blue
                        })
                        
                        # Close the polyline
                        polyline.close(True)
                    
                    # Save the document
                    doc.saveas(str(export_path))
                    
                    return FileResponse(
                        export_path,
                        media_type="application/dxf",
                        filename=filename
                    )
                except ImportError:
                    raise HTTPException(status_code=501, detail="DXF export requires ezdxf or dxf_exporter module")
        
        elif format.lower() == "svg":
            # Create SVG file
            svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width * scale}" height="{height * scale}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description>
      <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">{metadata['title']}</dc:title>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">{metadata['author']}</dc:creator>
      <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">{metadata['date']}</dc:date>
      <dc:units xmlns:dc="http://purl.org/dc/elements/1.1/">{units}</dc:units>
      <dc:scale xmlns:dc="http://purl.org/dc/elements/1.1/">{scale}</dc:scale>
    </rdf:Description>
  </rdf:RDF>
</metadata>
"""
            
            # Add each contour
            svg_content += '<g id="contours">\n'
            
            # Organize by outer/inner layers
            svg_content += '  <g id="outer-contours" fill="none" stroke="blue" stroke-width="0.5">\n'
            for contour_data in processed_contours:
                if not contour_data["is_inner"]:
                    contour = contour_data["points"]
                    path_data = "M "
                    for point in contour.reshape(-1, 2):
                        x, y = point
                        path_data += f"{x},{y} "
                    path_data += "Z"  # Close the path
                    
                    svg_content += f'    <path d="{path_data}" />\n'
            svg_content += '  </g>\n'
            
            if include_inner_contours:
                svg_content += '  <g id="inner-contours" fill="none" stroke="green" stroke-width="0.5">\n'
                for contour_data in processed_contours:
                    if contour_data["is_inner"]:
                        contour = contour_data["points"]
                        path_data = "M "
                        for point in contour.reshape(-1, 2):
                            x, y = point
                            path_data += f"{x},{y} "
                        path_data += "Z"  # Close the path
                        
                        svg_content += f'    <path d="{path_data}" />\n'
                svg_content += '  </g>\n'
            
            svg_content += '</g>\n'
            
            # Close SVG document
            svg_content += "</svg>"
            
            # Write SVG file
            with open(export_path, "w") as f:
                f.write(svg_content)
            
            return FileResponse(
                export_path,
                media_type="image/svg+xml",
                filename=filename
            )
            
        elif format.lower() == "step":
            try:
                # Check if OCC or PythonOCC is available for STEP export
                import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
                import OCC.Core.gp as gp
                import OCC.Core.STEPControl as STEPControl
                import OCC.Core.TopoDS as TopoDS
                
                # Initialize STEP writer
                writer = STEPControl.STEPControl_Writer()
                
                # Create compounds for inner and outer contours
                compound = TopoDS.TopoDS_Compound()
                builder = BRepBuilderAPI.BRepBuilderAPI_MakeWire()
                
                # Process contours
                for contour_data in processed_contours:
                    contour = contour_data["points"]
                    points = []
                    
                    # Convert contour points to OCC points
                    for point in contour:
                        x, y = point[0]
                        points.append(gp.gp_Pnt(float(x), float(y), 0.0))
                    
                    # Create edges and wire
                    wire = BRepBuilderAPI.BRepBuilderAPI_MakeWire()
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        edge = BRepBuilderAPI.BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                        wire.Add(edge)
                    
                    # Add wire to compound
                    builder.Add(wire.Wire())
                
                # Create the final compound
                compound = builder.Wire()
                
                # Write to STEP file
                writer.Transfer(compound, STEPControl.STEPControl_AsIs)
                writer.Write(str(export_path))
                
                return FileResponse(
                    export_path,
                    media_type="application/step",
                    filename=filename
                )
            except ImportError:
                raise HTTPException(status_code=501, detail="STEP export requires OCC or PythonOCC module")
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported CAD format: {format}")
    except Exception as e:
        logger.error(f"Error exporting to CAD format: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CAD export error: {str(e)}")

# Main entry point
if __name__ == "__main__":
    # Set up global segmentation settings
    segmentation_settings = {
        "quality_threshold": 0.8,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "apply_crf": True,
        "include_inner_contours": True,
        "hierarchical_export": True,
        "last_updated": time.time(),
        "user": "FETHl",
        "date": "2025-04-08 14:23:14"
    }
    
    # Start the API server
    logger.info("Starting API-SAM server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)