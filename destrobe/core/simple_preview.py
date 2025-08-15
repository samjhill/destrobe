"""
Simple preview generation that works with all filter methods including enhanced ones.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from destrobe.core.io import VideoCapture, VideoWriter, parse_time_string
from destrobe.core.filters import apply_filter
from destrobe.core.enhanced_filters import enhanced_apply_filter


def create_simple_preview(
    input_file: Path,
    output_file: Path,
    method: str = "median3",
    strength: float = 0.5,
    flash_thresh: float = 0.12,
    duration_seconds: int = 10,
    start_time: str = "00:00:00",
) -> None:
    """
    Create a side-by-side preview that works with all filter methods.
    
    Args:
        input_file: Path to input video file
        output_file: Path to output preview file
        method: Processing method to use
        strength: Filter strength
        flash_thresh: Flash detection threshold
        duration_seconds: Duration of preview in seconds
        start_time: Start time in HH:MM:SS format
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
        
        # Read all frames first
        print(f"Reading {total_frames} frames...")
        original_frames = []
        
        for i in range(total_frames):
            ret, frame = cap.read_frame()
            if not ret:
                break
            
            # Convert to float32 for processing
            frame_float = frame.astype(np.float32) / 255.0
            original_frames.append(frame_float)
        
        print(f"Processing {len(original_frames)} frames with {method}...")
        
        # Process frames with proper buffering
        processed_frames = []
        frame_buffer = []
        
        for i, frame in enumerate(tqdm(original_frames, desc="Processing frames")):
            frame_buffer.append(frame)
            
            # Keep appropriate buffer size
            if method.startswith("enhanced_"):
                max_buffer = 7
            else:
                max_buffer = 5
            
            if len(frame_buffer) > max_buffer:
                frame_buffer.pop(0)
            
            # Determine when to process based on method
            can_process = False
            if method in ["median3", "enhanced_median"]:
                can_process = len(frame_buffer) >= 3
            elif method.startswith("enhanced_"):
                can_process = len(frame_buffer) >= 3
            else:
                can_process = True  # EMA and flashcap can process immediately
            
            if can_process:
                try:
                    if method.startswith("enhanced_"):
                        processed_frame = enhanced_apply_filter(
                            frame_buffer, method, strength, flash_thresh
                        )
                    else:
                        processed_frame = apply_filter(
                            frame_buffer, method, strength, flash_thresh
                        )
                    processed_frames.append(processed_frame)
                except Exception as e:
                    print(f"Error processing frame {i}: {e}")
                    # Fallback to original frame
                    processed_frames.append(frame.copy())
            else:
                # For first few frames that can't be processed yet
                processed_frames.append(frame.copy())
        
        # Ensure we have the same number of frames
        min_frames = min(len(original_frames), len(processed_frames))
        original_frames = original_frames[:min_frames]
        processed_frames = processed_frames[:min_frames]
        
        print(f"Creating side-by-side video with {min_frames} frames...")
        
        # Create video writer for side-by-side output
        output_width = cap.width * 2 + 4  # Double width plus divider
        writer = VideoWriter(output_file, cap.fps, output_width, cap.height)
        
        try:
            # Create side-by-side frames
            for orig, proc in tqdm(zip(original_frames, processed_frames), 
                                   total=min_frames, desc="Compositing"):
                
                # Convert back to uint8
                orig_uint8 = (orig * 255).astype(np.uint8)
                proc_uint8 = (proc * 255).astype(np.uint8)
                
                # Create side-by-side frame
                combined = create_side_by_side_frame(orig_uint8, proc_uint8)
                
                # Add text labels
                combined = add_text_overlay(
                    combined, 
                    left_text="Original",
                    right_text=f"Processed ({method})"
                )
                
                # Write frame
                writer.write_frame(combined)
        
        finally:
            writer.close()
    
    finally:
        cap.close()


def create_side_by_side_frame(
    left_frame: np.ndarray, 
    right_frame: np.ndarray,
    divider_width: int = 4
) -> np.ndarray:
    """Create a side-by-side comparison frame."""
    height, width = left_frame.shape[:2]
    
    # Create output frame with double width plus divider
    output_width = width * 2 + divider_width
    output_frame = np.zeros((height, output_width, 3), dtype=left_frame.dtype)
    
    # Place left frame
    output_frame[:, :width] = left_frame
    
    # Create divider (white line)
    if divider_width > 0:
        output_frame[:, width:width + divider_width] = 255
    
    # Place right frame
    output_frame[:, width + divider_width:] = right_frame
    
    return output_frame


def add_text_overlay(
    frame: np.ndarray,
    left_text: str = "Original",
    right_text: str = "Processed",
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """Add text overlay to identify left and right sides."""
    frame_with_text = frame.copy()
    height, width = frame.shape[:2]
    
    # Define font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    # Calculate positions
    margin = 10
    y_pos = 30
    
    # Left text position
    left_x = margin
    
    # Right text position (second half of frame)
    half_width = width // 2
    right_x = half_width + margin
    
    # Draw text shadows
    cv2.putText(frame_with_text, left_text, (left_x + 1, y_pos + 1), 
                font, font_scale, shadow_color, thickness)
    cv2.putText(frame_with_text, right_text, (right_x + 1, y_pos + 1), 
                font, font_scale, shadow_color, thickness)
    
    # Draw actual text
    cv2.putText(frame_with_text, left_text, (left_x, y_pos), 
                font, font_scale, text_color, thickness)
    cv2.putText(frame_with_text, right_text, (right_x, y_pos), 
                font, font_scale, text_color, thickness)
    
    return frame_with_text
