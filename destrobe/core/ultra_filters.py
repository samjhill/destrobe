"""
Ultra-aggressive filtering algorithms for maximum flicker reduction.
For cases where standard algorithms still leave visible flashing.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from destrobe.core.filters import bgr_to_yuv, yuv_to_bgr


def ultra_flash_suppression(
    frames: List[np.ndarray],
    flash_thresh: float = 0.02,  # Very low threshold
    strength: float = 0.9,       # Very high strength
    temporal_window: int = 5     # Larger window
) -> np.ndarray:
    """
    Ultra-aggressive flash suppression for maximum flicker reduction.
    
    Args:
        frames: List of 5+ consecutive frames in float32 BGR format
        flash_thresh: Very low threshold for flash detection
        strength: Very high strength for flash suppression
        temporal_window: Number of frames to consider for suppression
    
    Returns:
        Heavily processed frame with maximum flash suppression
    """
    if len(frames) < 3:
        return frames[-1].copy()
    
    # Use middle frame as reference
    mid_idx = len(frames) // 2
    curr_frame = frames[mid_idx]
    
    # Extract Y components from all frames
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    
    # Multi-scale flash detection
    flash_detected = False
    
    # 1. Global brightness spike detection
    brightness_values = [np.mean(y) for y in y_components]
    curr_brightness = brightness_values[mid_idx]
    
    # Compare with all neighbors
    for i, brightness in enumerate(brightness_values):
        if i != mid_idx:
            if curr_brightness - brightness > flash_thresh:
                flash_detected = True
                break
    
    # 2. Local region flash detection (more sensitive)
    if not flash_detected:
        for i, y_frame in enumerate(y_components):
            if i != mid_idx:
                diff = np.abs(y_curr - y_frame)
                # If more than 10% of pixels have significant brightness change
                flash_pixels = np.sum(diff > flash_thresh) / diff.size
                if flash_pixels > 0.1:
                    flash_detected = True
                    break
    
    # 3. Temporal gradient analysis
    if not flash_detected and len(frames) >= 5:
        # Check for rapid brightness changes
        grad = np.gradient([np.mean(y) for y in y_components])
        if np.max(np.abs(grad)) > flash_thresh * 2:
            flash_detected = True
    
    if flash_detected:
        # Ultra-aggressive suppression
        
        # Method 1: Weighted temporal median with bias toward stable frames
        weights = []
        for i, y_frame in enumerate(y_components):
            if i == mid_idx:
                weights.append(0.1)  # Very low weight for current frame
            else:
                # Higher weight for frames with lower brightness difference
                diff = np.mean(np.abs(y_frame - y_curr))
                weight = 1.0 / (1.0 + diff * 10)
                weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average of all frames
        y_suppressed = np.zeros_like(y_curr)
        for i, (y_frame, weight) in enumerate(zip(y_components, weights)):
            y_suppressed += y_frame * weight
        
        # Method 2: Spatial smoothing to reduce remaining artifacts
        y_suppressed = cv2.GaussianBlur(y_suppressed, (3, 3), 0.5)
        
        # Method 3: Adaptive clamping based on neighborhood
        if len(frames) >= 3:
            # Clamp extreme values to neighborhood range
            neighbor_min = np.minimum(y_components[0], y_components[-1]) if mid_idx > 0 else y_components[-1]
            neighbor_max = np.maximum(y_components[0], y_components[-1]) if mid_idx > 0 else y_components[-1]
            
            # Expand the range slightly to avoid over-clamping
            range_expansion = (neighbor_max - neighbor_min) * 0.1
            neighbor_min -= range_expansion
            neighbor_max += range_expansion
            
            y_suppressed = np.clip(y_suppressed, neighbor_min, neighbor_max)
        
        y_filtered = y_suppressed
        
    else:
        # Even for non-flash frames, apply heavy temporal smoothing
        # This helps with micro-flickers and gradual changes
        temporal_strength = 0.4  # Much stronger than normal
        
        if len(frames) >= 5:
            # Use median of all frames for maximum stability
            y_stack = np.stack(y_components, axis=2)
            y_median = np.median(y_stack, axis=2)
            y_filtered = (1 - temporal_strength) * y_curr + temporal_strength * y_median
        else:
            # Use average of neighbors
            y_neighbors = np.mean([y for i, y in enumerate(y_components) if i != mid_idx], axis=0)
            y_filtered = (1 - temporal_strength) * y_curr + temporal_strength * y_neighbors
    
    # Final spatial smoothing for any remaining artifacts
    y_filtered = cv2.bilateralFilter(y_filtered.astype(np.float32), 5, 0.02, 0.02)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_filtered, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def ultra_temporal_stabilizer(
    frames: List[np.ndarray],
    stabilization_strength: float = 0.8
) -> np.ndarray:
    """
    Ultra-strong temporal stabilization using multiple techniques.
    
    Args:
        frames: List of frames in float32 BGR format
        stabilization_strength: Strength of stabilization (0.0-1.0)
    
    Returns:
        Heavily stabilized frame
    """
    if len(frames) < 3:
        return frames[-1].copy()
    
    mid_idx = len(frames) // 2
    curr_frame = frames[mid_idx]
    
    # Extract luminance components
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    
    # Method 1: Multi-frame temporal median
    y_stack = np.stack(y_components, axis=2)
    y_median = np.median(y_stack, axis=2)
    
    # Method 2: Exponential weighted average with distance weighting
    weights = []
    for i in range(len(frames)):
        distance = abs(i - mid_idx) + 1
        weight = 1.0 / distance
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    y_weighted = np.zeros_like(y_curr)
    for y_frame, weight in zip(y_components, weights):
        y_weighted += y_frame * weight
    
    # Method 3: Edge-preserving smoothing
    # Detect edges to preserve important details
    edges = cv2.Canny((y_curr * 255).astype(np.uint8), 50, 150)
    edge_mask = edges.astype(np.float32) / 255.0
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3)), iterations=1)
    
    # Combine methods with edge preservation
    y_stabilized = (
        stabilization_strength * 0.6 * y_median +
        stabilization_strength * 0.4 * y_weighted +
        (1 - stabilization_strength) * y_curr
    )
    
    # Preserve edges
    y_final = edge_mask * y_curr + (1 - edge_mask) * y_stabilized
    
    # Final bilateral filter for smoothness
    y_final = cv2.bilateralFilter(y_final.astype(np.float32), 7, 0.03, 0.03)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_final, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def ultra_apply_filter(
    frames: List[np.ndarray],
    method: str,
    strength: float = 0.9,
    flash_thresh: float = 0.015,
    **kwargs
) -> np.ndarray:
    """
    Apply ultra-aggressive filter methods for maximum flicker reduction.
    
    Args:
        frames: List of frames in float32 BGR format
        method: Filter method ("ultra_suppress", "ultra_stabilize")
        strength: Filter strength parameter (higher = more aggressive)
        flash_thresh: Flash detection threshold (lower = more sensitive)
        **kwargs: Additional method-specific parameters
    
    Returns:
        Heavily processed frame
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if method == "ultra_suppress":
        if len(frames) < 3:
            return frames[-1].copy()
        # Use up to 7 frames for maximum context
        frame_subset = frames[-7:] if len(frames) >= 7 else frames
        return ultra_flash_suppression(frame_subset, flash_thresh, strength, **kwargs)
    
    elif method == "ultra_stabilize":
        if len(frames) < 3:
            return frames[-1].copy()
        # Use up to 5 frames for stabilization
        frame_subset = frames[-5:] if len(frames) >= 5 else frames
        return ultra_temporal_stabilizer(frame_subset, strength, **kwargs)
    
    else:
        raise ValueError(f"Unknown ultra filter method: {method}")


def hybrid_ultra_filter(
    frames: List[np.ndarray],
    flash_thresh: float = 0.015,
    strength: float = 0.9
) -> np.ndarray:
    """
    Hybrid approach combining flash suppression and temporal stabilization.
    
    Args:
        frames: List of frames in float32 BGR format
        flash_thresh: Flash detection threshold
        strength: Overall filter strength
    
    Returns:
        Frame processed with hybrid ultra filtering
    """
    if len(frames) < 3:
        return frames[-1].copy()
    
    # First pass: Flash suppression
    suppressed = ultra_flash_suppression(frames, flash_thresh, strength * 0.8)
    
    # Second pass: Temporal stabilization on the suppressed result
    # Create new frame list with suppressed frame in the middle
    mid_idx = len(frames) // 2
    stabilize_frames = frames.copy()
    stabilize_frames[mid_idx] = suppressed
    
    final_result = ultra_temporal_stabilizer(stabilize_frames, strength * 0.6)
    
    return final_result
