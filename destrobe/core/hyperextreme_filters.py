"""
Hyper-extreme filters - the absolute maximum flicker reduction possible.
Quality will be significantly degraded but flicker will be virtually eliminated.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from destrobe.core.filters import bgr_to_yuv, yuv_to_bgr


def hyperextreme_flashcap(
    frames: List[np.ndarray],
    detection_threshold: float = 0.0005,  # Ultra-ultra-sensitive
    suppression_strength: float = 0.995,   # Near-complete suppression
    spatial_smoothing: float = 3.0         # Heavy spatial smoothing
) -> np.ndarray:
    """
    Hyper-extreme version of enhanced flashcap - maximum possible flicker reduction.
    
    Args:
        frames: List of 3+ consecutive frames in float32 BGR format
        detection_threshold: Ultra-low threshold for any brightness change
        suppression_strength: Near-total suppression (0.99+)
        spatial_smoothing: Heavy spatial smoothing strength
    
    Returns:
        Frame with hyper-extreme flash suppression
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
    
    # Hyper-extreme detection - ANY change triggers suppression
    flash_detected = False
    
    # 1. Pixel-level ultra-sensitive detection
    for i, y_frame in enumerate(y_components):
        if i != mid_idx:
            # Check mean difference
            mean_diff = abs(np.mean(y_curr) - np.mean(y_frame))
            if mean_diff > detection_threshold:
                flash_detected = True
                break
            
            # Check max difference
            max_diff = np.max(np.abs(y_curr - y_frame))
            if max_diff > detection_threshold * 10:
                flash_detected = True
                break
            
            # Check standard deviation change
            std_diff = abs(np.std(y_curr) - np.std(y_frame))
            if std_diff > detection_threshold * 5:
                flash_detected = True
                break
    
    # 2. Multi-scale detection
    if not flash_detected:
        # Check at different image scales
        for scale in [0.5, 0.25]:
            h, w = y_curr.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            y_curr_scaled = cv2.resize(y_curr, (new_w, new_h))
            
            for i, y_frame in enumerate(y_components):
                if i != mid_idx:
                    y_frame_scaled = cv2.resize(y_frame, (new_w, new_h))
                    diff = abs(np.mean(y_curr_scaled) - np.mean(y_frame_scaled))
                    
                    if diff > detection_threshold * scale:
                        flash_detected = True
                        break
            
            if flash_detected:
                break
    
    # Apply suppression (almost always triggered due to ultra-sensitivity)
    if flash_detected or True:  # Force suppression for maximum effect
        
        # Strategy 1: Multi-frame weighted median with extreme bias
        weights = []
        for i, y_frame in enumerate(y_components):
            if i == mid_idx:
                weights.append(1 - suppression_strength)  # Tiny weight for current
            else:
                # High weight for stable neighbors
                weights.append(suppression_strength / (len(y_components) - 1))
        
        y_weighted = np.zeros_like(y_curr)
        for y_frame, weight in zip(y_components, weights):
            y_weighted += y_frame * weight
        
        # Strategy 2: Extreme temporal clamping
        if len(frames) >= 3:
            # Find min/max of all other frames
            other_frames = [y for i, y in enumerate(y_components) if i != mid_idx]
            y_min = np.minimum.reduce(other_frames)
            y_max = np.maximum.reduce(other_frames)
            
            # Allow only tiny deviations from neighbor range
            range_expansion = (y_max - y_min) * 0.01  # 1% expansion
            y_weighted = np.clip(y_weighted, 
                               y_min - range_expansion,
                               y_max + range_expansion)
        
        # Strategy 3: Multi-stage spatial smoothing
        y_smoothed = y_weighted
        
        # Stage 1: Bilateral filter (edge-preserving but strong)
        y_smoothed = cv2.bilateralFilter(
            y_smoothed.astype(np.float32), 
            int(spatial_smoothing * 3), 
            spatial_smoothing * 0.02, 
            spatial_smoothing * 0.02
        )
        
        # Stage 2: Gaussian blur for final smoothness
        kernel_size = int(spatial_smoothing * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        y_smoothed = cv2.GaussianBlur(y_smoothed, (kernel_size, kernel_size), spatial_smoothing * 0.5)
        
        # Strategy 4: Temporal consistency enforcement
        # If we have more than 3 frames, enforce consistency across time
        if len(frames) >= 5:
            # Check if current result is consistent with temporal trend
            temporal_median = np.median(y_components, axis=0)
            consistency_weight = 0.3
            y_smoothed = (1 - consistency_weight) * y_smoothed + consistency_weight * temporal_median
        
        y_final = y_smoothed
    else:
        # Even for "stable" frames, apply heavy stabilization
        y_stack = np.stack(y_components, axis=2)
        y_median = np.median(y_stack, axis=2)
        stabilization = 0.8
        y_final = (1 - stabilization) * y_curr + stabilization * y_median
    
    # Final quality vs stability trade-off - prioritize stability
    y_final = cv2.GaussianBlur(y_final, (3, 3), 0.5)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_final, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def total_flicker_annihilation(
    frames: List[np.ndarray],
    annihilation_level: float = 0.999
) -> np.ndarray:
    """
    Total flicker annihilation - removes virtually all temporal variation.
    This will significantly degrade video quality but eliminate flicker completely.
    
    Args:
        frames: List of frames in float32 BGR format
        annihilation_level: How much temporal variation to destroy (0.995-0.999)
    
    Returns:
        Frame with total flicker annihilation
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
    
    # Total annihilation approach: Multiple stability measures combined
    
    # 1. Multi-frame median (most stable)
    y_stack = np.stack(y_components, axis=2)
    y_median = np.median(y_stack, axis=2)
    
    # 2. Stability-weighted average
    stability_weights = []
    for y_frame in y_components:
        # Frames closer to the median are more "stable"
        stability = 1.0 / (1.0 + np.mean(np.abs(y_frame - y_median)))
        stability_weights.append(stability)
    
    stability_weights = np.array(stability_weights)
    stability_weights = stability_weights / np.sum(stability_weights)
    
    y_stable = np.zeros_like(y_curr)
    for y_frame, weight in zip(y_components, stability_weights):
        y_stable += y_frame * weight
    
    # 3. Temporal consistency (running average simulation)
    # Simulate a running average that heavily dampens changes
    if len(frames) >= 7:
        # Use outer frames as "history"
        history_frames = y_components[:2] + y_components[-2:]
        y_history = np.mean(history_frames, axis=0)
        
        # Blend current with history using extreme dampening
        history_weight = 0.7
        y_stable = (1 - history_weight) * y_stable + history_weight * y_history
    
    # 4. Apply annihilation level
    y_annihilated = (1 - annihilation_level) * y_curr + annihilation_level * y_stable
    
    # 5. Multi-stage smoothing for absolute temporal stability
    
    # Stage 1: Heavy bilateral filtering
    y_annihilated = cv2.bilateralFilter(y_annihilated.astype(np.float32), 15, 0.1, 0.1)
    
    # Stage 2: Large Gaussian blur
    y_annihilated = cv2.GaussianBlur(y_annihilated, (11, 11), 3.0)
    
    # Stage 3: Morphological smoothing (opening + closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    y_annihilated = cv2.morphologyEx(y_annihilated, cv2.MORPH_OPEN, kernel)
    y_annihilated = cv2.morphologyEx(y_annihilated, cv2.MORPH_CLOSE, kernel)
    
    # 6. Final stability enforcement
    # Clamp to very narrow range based on frame history
    if len(frames) >= 5:
        frame_means = [np.mean(y) for y in y_components]
        safe_min = np.percentile(frame_means, 25)
        safe_max = np.percentile(frame_means, 75)
        
        # Global clamping
        global_mean = np.mean(y_annihilated)
        if global_mean < safe_min:
            y_annihilated = y_annihilated + (safe_min - global_mean)
        elif global_mean > safe_max:
            y_annihilated = y_annihilated - (global_mean - safe_max)
    
    # Reconstruct frame
    result = yuv_to_bgr(y_annihilated, u_curr, v_curr)
    return np.clip(result, 0.0, 1.0)


def hyperextreme_apply_filter(
    frames: List[np.ndarray],
    method: str,
    strength: float = 0.999,
    detection_thresh: float = 0.0005,
    **kwargs
) -> np.ndarray:
    """
    Apply hyper-extreme anti-flicker methods.
    
    Args:
        frames: List of frames in float32 BGR format
        method: Hyper-extreme method to use
        strength: Extreme strength parameter (0.995+)
        detection_thresh: Ultra-ultra-low detection threshold
        **kwargs: Additional method-specific parameters
    
    Returns:
        Frame with hyper-extreme flicker suppression
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if method == "hyperextreme":
        if len(frames) < 3:
            return frames[-1].copy()
        frame_subset = frames[-9:] if len(frames) >= 9 else frames
        return hyperextreme_flashcap(frame_subset, detection_thresh, strength, **kwargs)
    
    elif method == "annihilation":
        if len(frames) < 5:
            return frames[-1].copy()
        frame_subset = frames[-11:] if len(frames) >= 11 else frames
        return total_flicker_annihilation(frame_subset, strength, **kwargs)
    
    else:
        raise ValueError(f"Unknown hyper-extreme filter method: {method}")
