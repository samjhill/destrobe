"""
Enhanced filtering algorithms for destrobe with improved real-world performance.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from destrobe.core.filters import bgr_to_yuv, yuv_to_bgr


def enhanced_flash_detection(
    frames: List[np.ndarray],
    flash_thresh: float = 0.05,  # Much lower threshold
    strength: float = 0.5,
    use_percentile: bool = True
) -> np.ndarray:
    """
    Enhanced flash detection with improved sensitivity for real-world content.
    
    Args:
        frames: List of 3 consecutive frames in float32 BGR format
        flash_thresh: Lower threshold for flash detection
        strength: Filter strength for flash capping
        use_percentile: Use percentile-based detection for robustness
    
    Returns:
        Processed frame
    """
    if len(frames) != 3:
        raise ValueError("enhanced_flash_detection requires exactly 3 frames")
    
    prev_frame, curr_frame, next_frame = frames
    
    # Extract Y components
    y_prev, _, _ = bgr_to_yuv(prev_frame)
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    y_next, _, _ = bgr_to_yuv(next_frame)
    
    if use_percentile:
        # Use 90th percentile instead of mean for better flash detection
        bright_prev = np.percentile(y_prev, 90)
        bright_curr = np.percentile(y_curr, 90)
        bright_next = np.percentile(y_next, 90)
    else:
        bright_prev = np.mean(y_prev)
        bright_curr = np.mean(y_curr)
        bright_next = np.mean(y_next)
    
    # Enhanced flash detection: check both mean and max brightness increases
    delta_prev = bright_curr - bright_prev
    delta_next = bright_curr - bright_next
    
    # Also check for sudden brightness spikes in local regions
    diff_prev = np.abs(y_curr - y_prev)
    diff_next = np.abs(y_curr - y_next)
    local_spike = np.percentile(diff_prev, 95) > flash_thresh * 2
    
    flash_detected = (
        (delta_prev > flash_thresh and delta_next > flash_thresh) or
        (local_spike and delta_prev > flash_thresh * 0.5)
    )
    
    if flash_detected:
        # Enhanced flash capping with spatial awareness
        # Use minimum of neighbors but preserve detail
        y_min_neighbors = np.minimum(y_prev, y_next)
        
        # Adaptive blending based on local brightness
        local_brightness = cv2.GaussianBlur(y_curr, (5, 5), 1.0)
        adaptation = np.clip(local_brightness * 2, 0.3, 1.0)
        
        lambda_blend = 0.3 + 0.7 * strength * adaptation
        lambda_blend = np.clip(lambda_blend, 0.1, 0.9)
        
        y_filtered = lambda_blend * y_min_neighbors + (1.0 - lambda_blend) * y_curr
    else:
        # Mild temporal blend with motion compensation
        # Compute motion vectors using optical flow approximation
        flow_magnitude = np.mean(np.abs(y_curr - y_prev))
        
        if flow_magnitude > 0.02:  # High motion - less temporal smoothing
            temporal_strength = 0.1
        else:  # Low motion - more temporal smoothing
            temporal_strength = 0.2
        
        y_filtered = (1 - temporal_strength) * y_curr + \
                    temporal_strength * 0.5 * (y_prev + y_next)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_filtered, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def enhanced_median_filter(
    frames: List[np.ndarray],
    adaptive_window: bool = True
) -> np.ndarray:
    """
    Enhanced median filter with adaptive window size based on motion.
    
    Args:
        frames: List of 3 or 5 consecutive frames in float32 BGR format
        adaptive_window: Use motion-adaptive window selection
    
    Returns:
        Processed frame with enhanced median filtering
    """
    if len(frames) < 3:
        raise ValueError("enhanced_median_filter requires at least 3 frames")
    
    # Use middle frame as reference
    mid_idx = len(frames) // 2
    reference_frame = frames[mid_idx]
    
    # Extract Y components from all frames
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    if adaptive_window and len(frames) >= 5:
        # Compute motion for adaptive window selection
        y_ref = y_components[mid_idx]
        motion_scores = []
        
        for i, y_frame in enumerate(y_components):
            if i != mid_idx:
                motion = np.mean(np.abs(y_frame - y_ref))
                motion_scores.append(motion)
        
        avg_motion = np.mean(motion_scores)
        
        if avg_motion > 0.05:  # High motion - use smaller window
            # Use only 3 center frames
            start_idx = max(0, mid_idx - 1)
            end_idx = min(len(frames), mid_idx + 2)
            selected_frames = y_components[start_idx:end_idx]
        else:  # Low motion - use full window
            selected_frames = y_components
    else:
        selected_frames = y_components
    
    # Stack Y components and compute median
    y_stack = np.stack(selected_frames, axis=2)
    y_median = np.median(y_stack, axis=2)
    
    # Use chroma from reference frame
    _, u_ref, v_ref = bgr_to_yuv(reference_frame)
    
    # Reconstruct frame with median Y and original chroma
    result = yuv_to_bgr(y_median, u_ref, v_ref)
    return np.clip(result, 0.0, 1.0)


def enhanced_ema_filter(
    current_frame: np.ndarray, 
    previous_frame: Optional[np.ndarray],
    strength: float = 0.5,
    motion_compensation: bool = True
) -> np.ndarray:
    """
    Enhanced EMA filter with better motion compensation and edge preservation.
    
    Args:
        current_frame: Current frame in float32 BGR format
        previous_frame: Previous frame in float32 BGR format (None for first frame)
        strength: Filter strength parameter
        motion_compensation: Enable motion-aware filtering
    
    Returns:
        Processed frame
    """
    if previous_frame is None:
        return current_frame.copy()
    
    # Convert to YUV
    y_curr, u_curr, v_curr = bgr_to_yuv(current_frame)
    y_prev, _, _ = bgr_to_yuv(previous_frame)
    
    if motion_compensation:
        # Enhanced motion estimation using gradients
        grad_curr_x = cv2.Sobel(y_curr, cv2.CV_32F, 1, 0, ksize=3)
        grad_curr_y = cv2.Sobel(y_curr, cv2.CV_32F, 0, 1, ksize=3)
        grad_prev_x = cv2.Sobel(y_prev, cv2.CV_32F, 1, 0, ksize=3)
        grad_prev_y = cv2.Sobel(y_prev, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude difference
        grad_diff = np.sqrt((grad_curr_x - grad_prev_x)**2 + (grad_curr_y - grad_prev_y)**2)
        motion_map = cv2.GaussianBlur(grad_diff, (5, 5), 1.0)
        
        # Normalize motion map
        motion_normalized = np.clip(motion_map / 0.1, 0.0, 1.0)
        
        # Adaptive alpha based on local motion
        base_alpha = 0.2 + 0.6 * motion_normalized
        alpha = base_alpha * (1.0 - strength * 0.8)
        alpha = np.clip(alpha, 0.05, 0.95)
    else:
        # Simple global motion estimation
        motion = np.mean(np.abs(y_curr - y_prev))
        motion_normalized = np.clip(motion / 0.2, 0.0, 1.0)
        alpha = (0.25 + 0.55 * motion_normalized) * (1.0 - strength * 0.7)
        alpha = np.clip(alpha, 0.1, 0.9)
    
    # Apply EMA with spatial-varying alpha
    if motion_compensation:
        y_filtered = alpha * y_curr + (1.0 - alpha) * y_prev
    else:
        y_filtered = alpha * y_curr + (1.0 - alpha) * y_prev
    
    # Edge preservation using bilateral filter concept
    edge_strength = cv2.Sobel(y_curr, cv2.CV_32F, 1, 0, ksize=3)**2 + \
                   cv2.Sobel(y_curr, cv2.CV_32F, 0, 1, ksize=3)**2
    edge_mask = edge_strength > np.percentile(edge_strength, 90)
    
    # Preserve edges by using less temporal filtering
    y_filtered = np.where(edge_mask, 
                         0.8 * y_curr + 0.2 * y_prev,  # Less filtering on edges
                         y_filtered)  # Normal filtering elsewhere
    
    # Reconstruct frame
    result = yuv_to_bgr(y_filtered, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def enhanced_apply_filter(
    frames: List[np.ndarray],
    method: str,
    strength: float = 0.5,
    flash_thresh: float = 0.05,  # Lower default threshold
    **kwargs
) -> np.ndarray:
    """
    Apply enhanced filter methods with improved real-world performance.
    
    Args:
        frames: List of frames in float32 BGR format
        method: Filter method ("enhanced_median", "enhanced_ema", "enhanced_flashcap")
        strength: Filter strength parameter
        flash_thresh: Flash detection threshold (lower for more sensitivity)
        **kwargs: Additional method-specific parameters
    
    Returns:
        Processed frame
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if method == "enhanced_median":
        if len(frames) < 3:
            return frames[-1].copy()
        return enhanced_median_filter(frames[-5:] if len(frames) >= 5 else frames[-3:], **kwargs)
    
    elif method == "enhanced_ema":
        current = frames[-1]
        previous = frames[-2] if len(frames) >= 2 else None
        return enhanced_ema_filter(current, previous, strength, **kwargs)
    
    elif method == "enhanced_flashcap":
        if len(frames) < 3:
            return frames[-1].copy()
        return enhanced_flash_detection(frames[-3:], flash_thresh, strength, **kwargs)
    
    else:
        raise ValueError(f"Unknown enhanced filter method: {method}")
