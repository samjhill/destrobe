"""
Extreme anti-flicker algorithms - maximum flicker suppression at any cost.
For cases where even enhanced methods leave visible strobing.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from destrobe.core.filters import bgr_to_yuv, yuv_to_bgr


def extreme_temporal_smoothing(
    frames: List[np.ndarray],
    smoothing_strength: float = 0.95,
    temporal_window: int = 7
) -> np.ndarray:
    """
    Extreme temporal smoothing - prioritizes stability over everything else.
    
    Args:
        frames: List of frames in float32 BGR format
        smoothing_strength: How aggressively to smooth (0.9-0.99)
        temporal_window: Number of frames to average
    
    Returns:
        Heavily smoothed frame
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
    
    # Method 1: Heavy temporal averaging with exponential weighting
    weights = []
    for i in range(len(frames)):
        # Exponential decay from center
        distance = abs(i - mid_idx)
        weight = np.exp(-distance * 0.5)
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Weighted average of all frames
    y_averaged = np.zeros_like(y_curr)
    for y_frame, weight in zip(y_components, weights):
        y_averaged += y_frame * weight
    
    # Method 2: Apply heavy smoothing strength
    y_smoothed = (1 - smoothing_strength) * y_curr + smoothing_strength * y_averaged
    
    # Method 3: Additional spatial smoothing to eliminate micro-variations
    y_smoothed = cv2.GaussianBlur(y_smoothed, (5, 5), 1.0)
    
    # Method 4: Clamp extreme variations
    # Find the range of neighbor frames
    neighbor_frames = [y for i, y in enumerate(y_components) if i != mid_idx]
    if neighbor_frames:
        min_neighbor = np.minimum.reduce(neighbor_frames)
        max_neighbor = np.maximum.reduce(neighbor_frames)
        
        # Expand range slightly but clamp extreme deviations
        range_expansion = (max_neighbor - min_neighbor) * 0.05
        y_smoothed = np.clip(y_smoothed, 
                           min_neighbor - range_expansion,
                           max_neighbor + range_expansion)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_smoothed, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def extreme_flash_elimination(
    frames: List[np.ndarray],
    detection_threshold: float = 0.005,  # Ultra-sensitive
    suppression_strength: float = 0.98    # Near-total suppression
) -> np.ndarray:
    """
    Extreme flash elimination - detects and eliminates even tiny flickers.
    
    Args:
        frames: List of frames in float32 BGR format
        detection_threshold: Ultra-low threshold for detection
        suppression_strength: Near-total suppression strength
    
    Returns:
        Frame with extreme flash suppression
    """
    if len(frames) < 3:
        return frames[-1].copy()
    
    mid_idx = len(frames) // 2
    curr_frame = frames[mid_idx]
    
    # Extract Y components
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    
    # Ultra-sensitive flash detection using multiple metrics
    flash_detected = False
    
    # 1. Global brightness spike detection (ultra-sensitive)
    curr_mean = np.mean(y_curr)
    for i, y_frame in enumerate(y_components):
        if i != mid_idx:
            frame_mean = np.mean(y_frame)
            if abs(curr_mean - frame_mean) > detection_threshold:
                flash_detected = True
                break
    
    # 2. Percentile-based detection (catches localized flashes)
    if not flash_detected:
        curr_p95 = np.percentile(y_curr, 95)
        for i, y_frame in enumerate(y_components):
            if i != mid_idx:
                frame_p95 = np.percentile(y_frame, 95)
                if abs(curr_p95 - frame_p95) > detection_threshold * 3:
                    flash_detected = True
                    break
    
    # 3. Local region variance detection
    if not flash_detected:
        # Divide image into blocks and check for variance spikes
        h, w = y_curr.shape
        block_size = 32
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                curr_block = y_curr[y:y+block_size, x:x+block_size]
                curr_var = np.var(curr_block)
                
                for i, y_frame in enumerate(y_components):
                    if i != mid_idx:
                        frame_block = y_frame[y:y+block_size, x:x+block_size]
                        frame_var = np.var(frame_block)
                        
                        if abs(curr_var - frame_var) > detection_threshold * 0.1:
                            flash_detected = True
                            break
                
                if flash_detected:
                    break
            if flash_detected:
                break
    
    if flash_detected:
        # Extreme suppression: replace almost entirely with stable average
        
        # Create stable reference from all other frames
        other_frames = [y for i, y in enumerate(y_components) if i != mid_idx]
        stable_reference = np.median(other_frames, axis=0)
        
        # Apply extreme suppression
        y_suppressed = (1 - suppression_strength) * y_curr + suppression_strength * stable_reference
        
        # Additional smoothing for suppressed areas
        y_suppressed = cv2.bilateralFilter(y_suppressed.astype(np.float32), 9, 0.05, 0.05)
        
    else:
        # Even for "non-flash" frames, apply heavy stabilization
        stabilization_strength = 0.7
        
        # Use median of all frames for maximum stability
        y_stack = np.stack(y_components, axis=2)
        y_median = np.median(y_stack, axis=2)
        
        y_suppressed = (1 - stabilization_strength) * y_curr + stabilization_strength * y_median
    
    # Final extreme smoothing pass
    y_suppressed = cv2.GaussianBlur(y_suppressed, (7, 7), 1.5)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_suppressed, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def nuclear_flicker_destroyer(
    frames: List[np.ndarray],
    destruction_level: float = 0.99
) -> np.ndarray:
    """
    Nuclear option - destroys virtually all temporal variation.
    Quality will be significantly degraded but flicker will be eliminated.
    
    Args:
        frames: List of frames in float32 BGR format
        destruction_level: How much temporal variation to destroy (0.95-0.99)
    
    Returns:
        Frame with nuclear-level flicker destruction
    """
    if len(frames) < 5:
        return frames[-1].copy()
    
    mid_idx = len(frames) // 2
    curr_frame = frames[mid_idx]
    
    # Extract Y components
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    
    # Nuclear approach: Multi-stage temporal destruction
    
    # Stage 1: Heavy temporal median
    y_stack = np.stack(y_components, axis=2)
    y_median = np.median(y_stack, axis=2)
    
    # Stage 2: Extreme weighted average (heavily favor stable frames)
    weights = []
    for i, y_frame in enumerate(y_components):
        # Compute "stability score" - how different this frame is from median
        diff_from_median = np.mean(np.abs(y_frame - y_median))
        # More stable frames get exponentially higher weight
        stability_weight = np.exp(-diff_from_median * 20)
        weights.append(stability_weight)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    y_weighted = np.zeros_like(y_curr)
    for y_frame, weight in zip(y_components, weights):
        y_weighted += y_frame * weight
    
    # Stage 3: Combine median and weighted with extreme bias toward stability
    y_nuclear = 0.3 * y_median + 0.7 * y_weighted
    
    # Stage 4: Apply destruction level
    y_destroyed = (1 - destruction_level) * y_curr + destruction_level * y_nuclear
    
    # Stage 5: Heavy spatial smoothing to eliminate any remaining artifacts
    y_destroyed = cv2.bilateralFilter(y_destroyed.astype(np.float32), 11, 0.08, 0.08)
    
    # Stage 6: Final Gaussian blur for ultimate smoothness
    y_destroyed = cv2.GaussianBlur(y_destroyed, (9, 9), 2.0)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_destroyed, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def extreme_apply_filter(
    frames: List[np.ndarray],
    method: str,
    strength: float = 0.98,
    detection_thresh: float = 0.002,
    **kwargs
) -> np.ndarray:
    """
    Apply extreme anti-flicker methods.
    
    Args:
        frames: List of frames in float32 BGR format
        method: Extreme method to use
        strength: Extreme strength parameter
        detection_thresh: Ultra-low detection threshold
        **kwargs: Additional method-specific parameters
    
    Returns:
        Frame with extreme flicker suppression
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if method == "extreme_smooth":
        if len(frames) < 3:
            return frames[-1].copy()
        frame_subset = frames[-7:] if len(frames) >= 7 else frames
        return extreme_temporal_smoothing(frame_subset, strength, **kwargs)
    
    elif method == "extreme_flash":
        if len(frames) < 3:
            return frames[-1].copy()
        frame_subset = frames[-7:] if len(frames) >= 7 else frames
        return extreme_flash_elimination(frame_subset, detection_thresh, strength, **kwargs)
    
    elif method == "nuclear":
        if len(frames) < 5:
            return frames[-1].copy()
        frame_subset = frames[-9:] if len(frames) >= 9 else frames
        return nuclear_flicker_destroyer(frame_subset, strength, **kwargs)
    
    else:
        raise ValueError(f"Unknown extreme filter method: {method}")
