"""
Preview functionality for destrobe - side-by-side comparison video generation.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from destrobe.core.filters import FrameBuffer
from destrobe.core.io import VideoCapture, VideoWriter, parse_time_string


def create_side_by_side_frame(
    left_frame: np.ndarray, 
    right_frame: np.ndarray,
    divider_width: int = 4
) -> np.ndarray:
    """
    Create a side-by-side comparison frame.
    
    Args:
        left_frame: Left frame (original)
        right_frame: Right frame (processed)
        divider_width: Width of the divider line in pixels
    
    Returns:
        Combined frame with left and right side by side
    """
    height, width = left_frame.shape[:2]
    
    # Create output frame with double width plus divider
    output_width = width * 2 + divider_width
    output_frame = np.zeros((height, output_width, 3), dtype=left_frame.dtype)
    
    # Place left frame
    output_frame[:, :width] = left_frame
    
    # Create divider (white line)
    if divider_width > 0:
        divider_color = (255, 255, 255) if left_frame.dtype == np.uint8 else (1.0, 1.0, 1.0)
        output_frame[:, width:width + divider_width] = divider_color
    
    # Place right frame
    output_frame[:, width + divider_width:] = right_frame
    
    return output_frame


def add_text_overlay(
    frame: np.ndarray,
    left_text: str = "Original",
    right_text: str = "Processed",
    font_scale: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """
    Add text overlay to identify left and right sides.
    
    Args:
        frame: Input frame to add text to
        left_text: Text for left side
        right_text: Text for right side
        font_scale: Font scale factor
        thickness: Text thickness
    
    Returns:
        Frame with text overlay
    """
    frame_with_text = frame.copy()
    height, width = frame.shape[:2]
    
    # Define font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255) if frame.dtype == np.uint8 else (1.0, 1.0, 1.0)
    shadow_color = (0, 0, 0) if frame.dtype == np.uint8 else (0.0, 0.0, 0.0)
    
    # Calculate text sizes
    (left_w, left_h), _ = cv2.getTextSize(left_text, font, font_scale, thickness)
    (right_w, right_h), _ = cv2.getTextSize(right_text, font, font_scale, thickness)
    
    # Position text
    margin = 20
    y_pos = margin + left_h
    
    # Left text position (accounting for possible divider)
    left_x = margin
    
    # Right text position (second half of frame)
    half_width = width // 2
    right_x = half_width + margin
    
    # Draw text shadows
    cv2.putText(frame_with_text, left_text, (left_x + 2, y_pos + 2), 
                font, font_scale, shadow_color, thickness)
    cv2.putText(frame_with_text, right_text, (right_x + 2, y_pos + 2), 
                font, font_scale, shadow_color, thickness)
    
    # Draw actual text
    cv2.putText(frame_with_text, left_text, (left_x, y_pos), 
                font, font_scale, text_color, thickness)
    cv2.putText(frame_with_text, right_text, (right_x, y_pos), 
                font, font_scale, text_color, thickness)
    
    return frame_with_text


def create_preview(
    input_file: Path,
    output_file: Path,
    method: str = "median3",
    strength: float = 0.5,
    flash_thresh: float = 0.12,
    duration_seconds: int = 10,
    start_time: str = "00:00:30",
    add_labels: bool = True
) -> None:
    """
    Create a side-by-side preview video comparing original and processed frames.
    
    Args:
        input_file: Path to input video file
        output_file: Path to output preview file
        method: Processing method to use
        strength: Filter strength
        flash_thresh: Flash detection threshold
        duration_seconds: Duration of preview in seconds
        start_time: Start time in HH:MM:SS format
        add_labels: Whether to add "Original"/"Processed" labels
    """
    
    # Open input video
    cap = VideoCapture(input_file)
    
    try:
        # Parse start time and calculate frame range
        start_seconds = parse_time_string(start_time)
        start_frame = int(start_seconds * cap.fps)
        end_frame = start_frame + int(duration_seconds * cap.fps)
        end_frame = min(end_frame, cap.frame_count)
        
        total_frames = end_frame - start_frame
        
        if total_frames <= 0:
            raise ValueError("No frames to process in the specified time range")
        
        # Seek to start position
        cap.seek_to_frame(start_frame)
        
        # Create video writer for side-by-side output
        # Output width is double the input width plus divider
        output_width = cap.width * 2 + 4
        writer = VideoWriter(output_file, cap.fps, output_width, cap.height)
        
        try:
            # Initialize frame buffer for filtering
            frame_buffer = FrameBuffer(method)
            
            # Process frames
            with tqdm(total=total_frames, desc="Creating preview") as pbar:
                processed_frames = []
                original_frames = []
                
                for i in range(total_frames):
                    ret, frame = cap.read_frame()
                    if not ret:
                        break
                    
                    # Convert to float32 for processing
                    frame_float = frame.astype(np.float32) / 255.0
                    original_frames.append(frame_float.copy())
                    
                    # Apply filter
                    filtered_frame = frame_buffer.add_frame(frame_float)
                    
                    if filtered_frame is not None:
                        processed_frames.append(filtered_frame)
                    elif i == 0:
                        # First frame for methods that need buffering
                        processed_frames.append(frame_float.copy())
                    
                    pbar.update(1)
                
                # Handle remaining frames from buffer
                remaining = frame_buffer.flush()
                processed_frames.extend(remaining)
                
                # Ensure we have the same number of original and processed frames
                min_frames = min(len(original_frames), len(processed_frames))
                original_frames = original_frames[:min_frames]
                processed_frames = processed_frames[:min_frames]
                
                # Create side-by-side frames
                with tqdm(total=min_frames, desc="Compositing preview") as pbar:
                    for orig, proc in zip(original_frames, processed_frames):
                        # Convert back to uint8
                        orig_uint8 = (orig * 255).astype(np.uint8)
                        proc_uint8 = (proc * 255).astype(np.uint8)
                        
                        # Create side-by-side frame
                        combined = create_side_by_side_frame(orig_uint8, proc_uint8)
                        
                        # Add text labels if requested
                        if add_labels:
                            combined = add_text_overlay(
                                combined, 
                                left_text="Original",
                                right_text=f"Processed ({method})"
                            )
                        
                        # Write frame
                        writer.write_frame(combined)
                        pbar.update(1)
        
        finally:
            writer.close()
    
    finally:
        cap.close()


def create_comparison_grid(
    input_file: Path,
    output_file: Path,
    methods: list = None,
    strength: float = 0.5,
    flash_thresh: float = 0.12,
    duration_seconds: int = 10,
    start_time: str = "00:00:30"
) -> None:
    """
    Create a 2x2 grid comparison of different processing methods.
    
    Args:
        input_file: Path to input video file
        output_file: Path to output comparison file
        methods: List of methods to compare (max 3, plus original)
        strength: Filter strength
        flash_thresh: Flash detection threshold
        duration_seconds: Duration of preview in seconds
        start_time: Start time in HH:MM:SS format
    """
    
    if methods is None:
        methods = ["median3", "ema", "flashcap"]
    
    methods = methods[:3]  # Limit to 3 methods
    
    # Open input video
    cap = VideoCapture(input_file)
    
    try:
        # Parse start time and calculate frame range
        start_seconds = parse_time_string(start_time)
        start_frame = int(start_seconds * cap.fps)
        end_frame = start_frame + int(duration_seconds * cap.fps)
        end_frame = min(end_frame, cap.frame_count)
        
        total_frames = end_frame - start_frame
        
        if total_frames <= 0:
            raise ValueError("No frames to process in the specified time range")
        
        # Seek to start position
        cap.seek_to_frame(start_frame)
        
        # Calculate grid dimensions
        cell_width = cap.width // 2
        cell_height = cap.height // 2
        
        # Create video writer for grid output
        writer = VideoWriter(output_file, cap.fps, cap.width, cap.height)
        
        try:
            # Initialize frame buffers for each method
            buffers = {method: FrameBuffer(method) for method in methods}
            
            # Process frames
            with tqdm(total=total_frames, desc="Creating comparison grid") as pbar:
                original_frames = []
                processed_frames = {method: [] for method in methods}
                
                for i in range(total_frames):
                    ret, frame = cap.read_frame()
                    if not ret:
                        break
                    
                    # Convert to float32 for processing
                    frame_float = frame.astype(np.float32) / 255.0
                    original_frames.append(frame_float.copy())
                    
                    # Apply each filter method
                    for method in methods:
                        filtered = buffers[method].add_frame(frame_float.copy())
                        if filtered is not None:
                            processed_frames[method].append(filtered)
                        elif i == 0:
                            processed_frames[method].append(frame_float.copy())
                    
                    pbar.update(1)
                
                # Handle remaining frames from buffers
                for method in methods:
                    remaining = buffers[method].flush()
                    processed_frames[method].extend(remaining)
                
                # Create grid frames
                min_frames = min([len(original_frames)] + 
                               [len(processed_frames[m]) for m in methods])
                
                with tqdm(total=min_frames, desc="Compositing grid") as pbar:
                    for i in range(min_frames):
                        # Get frames for this iteration
                        orig = original_frames[i]
                        
                        # Resize frames to fit in grid
                        orig_resized = cv2.resize(orig, (cell_width, cell_height))
                        
                        # Create 2x2 grid
                        grid = np.zeros((cap.height, cap.width, 3), dtype=np.float32)
                        
                        # Top-left: Original
                        grid[:cell_height, :cell_width] = orig_resized
                        
                        # Fill other cells with processed versions
                        positions = [
                            (0, cell_width),  # Top-right
                            (cell_height, 0),  # Bottom-left
                            (cell_height, cell_width)  # Bottom-right
                        ]
                        
                        for j, method in enumerate(methods):
                            if j < len(positions) and i < len(processed_frames[method]):
                                proc = processed_frames[method][i]
                                proc_resized = cv2.resize(proc, (cell_width, cell_height))
                                
                                y, x = positions[j]
                                grid[y:y+cell_height, x:x+cell_width] = proc_resized
                        
                        # Convert to uint8 and write
                        grid_uint8 = (grid * 255).astype(np.uint8)
                        
                        # Add method labels
                        labels = ["Original"] + methods
                        label_positions = [(10, 30), (cell_width + 10, 30), 
                                         (10, cell_height + 30), (cell_width + 10, cell_height + 30)]
                        
                        for label, pos in zip(labels, label_positions):
                            cv2.putText(grid_uint8, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (255, 255, 255), 2)
                        
                        writer.write_frame(grid_uint8)
                        pbar.update(1)
        
        finally:
            writer.close()
    
    finally:
        cap.close()
