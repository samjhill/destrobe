"""
Core filtering algorithms for destrobe - temporal video processing to reduce flicker.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def bgr_to_yuv(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert BGR frame to YUV components.
    
    Args:
        frame: Input frame in BGR format (float32, 0-1 range)
    
    Returns:
        Tuple of (Y, U, V) components
    """
    # Convert BGR to YUV using OpenCV
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return y, u, v


def yuv_to_bgr(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert YUV components back to BGR frame.
    
    Args:
        y: Y (luminance) component
        u: U (chroma) component  
        v: V (chroma) component
    
    Returns:
        BGR frame
    """
    yuv = cv2.merge([y, u, v])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr


def median3_filter(frames: List[np.ndarray]) -> np.ndarray:
    """
    Apply 3-frame temporal median filter on luminance.
    
    Args:
        frames: List of 3 consecutive frames in float32 BGR format
    
    Returns:
        Processed frame with median-filtered luminance
    """
    if len(frames) != 3:
        raise ValueError("median3_filter requires exactly 3 frames")
    
    # Extract Y components from all frames
    y_components = []
    for frame in frames:
        y, _, _ = bgr_to_yuv(frame)
        y_components.append(y)
    
    # Stack Y components and compute median
    y_stack = np.stack(y_components, axis=2)
    y_median = np.median(y_stack, axis=2)
    
    # Use chroma from center frame to avoid color smearing
    _, u_center, v_center = bgr_to_yuv(frames[1])
    
    # Reconstruct frame with median Y and original chroma
    result = yuv_to_bgr(y_median, u_center, v_center)
    
    # Clamp to valid range
    return np.clip(result, 0.0, 1.0)


def ema_filter(
    current_frame: np.ndarray, 
    previous_frame: Optional[np.ndarray],
    strength: float = 0.5
) -> np.ndarray:
    """
    Apply exponential moving average filter with motion-aware alpha.
    
    Args:
        current_frame: Current frame in float32 BGR format
        previous_frame: Previous frame in float32 BGR format (None for first frame)
        strength: Filter strength (0.0 = no filtering, 1.0 = maximum smoothing)
    
    Returns:
        Processed frame
    """
    if previous_frame is None:
        return current_frame.copy()
    
    # Convert to grayscale for motion detection
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute motion proxy (mean absolute difference)
    motion = np.mean(np.abs(current_gray - previous_gray))
    
    # Map motion to alpha: less motion = more smoothing
    # Motion range is roughly 0-0.5 for typical videos
    motion_normalized = np.clip(motion / 0.2, 0.0, 1.0)
    base_alpha = 0.25 + 0.55 * motion_normalized  # Range: 0.25 to 0.8
    
    # Apply strength: higher strength = lower alpha (more smoothing)
    alpha = base_alpha * (1.0 - strength * 0.7)
    alpha = np.clip(alpha, 0.1, 0.9)
    
    # Extract Y, U, V components
    y_curr, u_curr, v_curr = bgr_to_yuv(current_frame)
    y_prev, _, _ = bgr_to_yuv(previous_frame)
    
    # Apply EMA to Y component only
    y_filtered = alpha * y_curr + (1.0 - alpha) * y_prev
    
    # Keep current chroma
    result = yuv_to_bgr(y_filtered, u_curr, v_curr)
    
    return np.clip(result, 0.0, 1.0)


def flashcap_filter(
    frames: List[np.ndarray],
    flash_thresh: float = 0.12,
    strength: float = 0.5
) -> np.ndarray:
    """
    Apply flash detection and capping filter.
    
    Args:
        frames: List of 3 consecutive frames in float32 BGR format
        flash_thresh: Threshold for flash detection
        strength: Filter strength for flash capping
    
    Returns:
        Processed frame
    """
    if len(frames) != 3:
        raise ValueError("flashcap_filter requires exactly 3 frames")
    
    prev_frame, curr_frame, next_frame = frames
    
    # Extract Y components
    y_prev, _, _ = bgr_to_yuv(prev_frame)
    y_curr, u_curr, v_curr = bgr_to_yuv(curr_frame)
    y_next, _, _ = bgr_to_yuv(next_frame)
    
    # Compute mean luminance values
    mean_prev = np.mean(y_prev)
    mean_curr = np.mean(y_curr)
    mean_next = np.mean(y_next)
    
    # Flash detection: current frame is significantly brighter than both neighbors
    flash_detected = (
        (mean_curr - mean_prev) > flash_thresh and 
        (mean_curr - mean_next) > flash_thresh
    )
    
    if flash_detected:
        # Flash capping: blend with minimum of neighbors
        y_min_neighbors = np.minimum(y_prev, y_next)
        lambda_blend = 0.5 + 0.5 * strength
        y_filtered = lambda_blend * y_min_neighbors + (1.0 - lambda_blend) * y_curr
    else:
        # Mild temporal blend to reduce micro-flicker
        y_filtered = 0.7 * y_curr + 0.15 * y_prev + 0.15 * y_next
    
    # Reconstruct frame
    result = yuv_to_bgr(y_filtered, u_curr, v_curr)
    
    return np.clip(result, 0.0, 1.0)


class FrameBuffer:
    """Helper class to manage frame buffering for different filter types."""
    
    def __init__(self, filter_type: str, buffer_size: int = 3) -> None:
        self.filter_type = filter_type
        self.buffer_size = buffer_size
        self.frames: List[np.ndarray] = []
        self.previous_filtered: Optional[np.ndarray] = None
    
    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Add a frame to the buffer and return a filtered frame if ready.
        
        Args:
            frame: Input frame in float32 BGR format
        
        Returns:
            Filtered frame if buffer is ready, None otherwise
        """
        self.frames.append(frame.copy())
        
        # Keep buffer size manageable
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
        
        # Check if we can process a frame
        if self.filter_type == "median3":
            if len(self.frames) >= 3:
                # Process the middle frame
                filtered = median3_filter(self.frames[-3:])
                return filtered
        
        elif self.filter_type == "ema":
            filtered = ema_filter(frame, self.previous_filtered)
            self.previous_filtered = filtered.copy()
            return filtered
        
        elif self.filter_type == "flashcap":
            if len(self.frames) >= 3:
                # Process the middle frame
                filtered = flashcap_filter(self.frames[-3:])
                return filtered
        
        return None
    
    def flush(self) -> List[np.ndarray]:
        """
        Flush remaining frames from buffer.
        
        Returns:
            List of remaining processed frames
        """
        remaining = []
        
        if self.filter_type == "median3":
            # For remaining frames, just return them as-is or duplicate the last processed
            while len(self.frames) > 0:
                frame = self.frames.pop(0)
                remaining.append(frame)
        
        return remaining


def apply_filter(
    frames: List[np.ndarray],
    method: str,
    strength: float = 0.5,
    flash_thresh: float = 0.12
) -> np.ndarray:
    """
    Apply the specified filter method to a frame or frames.
    
    Args:
        frames: List of frames in float32 BGR format
        method: Filter method ("median3", "ema", "flashcap")
        strength: Filter strength parameter
        flash_thresh: Flash detection threshold
    
    Returns:
        Processed frame
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if method == "median3":
        if len(frames) < 3:
            # Not enough frames, return the current frame
            return frames[-1].copy()
        return median3_filter(frames[-3:])
    
    elif method == "ema":
        current = frames[-1]
        previous = frames[-2] if len(frames) >= 2 else None
        return ema_filter(current, previous, strength)
    
    elif method == "flashcap":
        if len(frames) < 3:
            # Not enough frames, return the current frame
            return frames[-1].copy()
        return flashcap_filter(frames[-3:], flash_thresh, strength)
    
    else:
        raise ValueError(f"Unknown filter method: {method}")


# Pre-allocated buffers for performance optimization
_temp_arrays = {}


def get_temp_array(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Get a temporary array for processing, reusing buffers when possible."""
    key = (shape, dtype)
    if key not in _temp_arrays:
        _temp_arrays[key] = np.empty(shape, dtype=dtype)
    return _temp_arrays[key]


def clear_temp_arrays() -> None:
    """Clear temporary array cache to free memory."""
    global _temp_arrays
    _temp_arrays.clear()
