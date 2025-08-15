"""
Metrics computation for destrobe - flicker detection and quality assessment.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# VideoCapture will be imported locally to avoid circular imports


def compute_luma_delta_sequence(frames: List[np.ndarray]) -> List[float]:
    """
    Compute frame-to-frame luminance delta sequence.
    
    Args:
        frames: List of frames in float32 BGR format
    
    Returns:
        List of absolute luminance deltas between consecutive frames
    """
    if len(frames) < 2:
        return []
    
    deltas = []
    prev_luma = None
    
    for frame in frames:
        # Convert to grayscale and compute mean luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_luma = np.mean(gray)
        
        if prev_luma is not None:
            delta = abs(current_luma - prev_luma)
            deltas.append(delta)
        
        prev_luma = current_luma
    
    return deltas


def compute_flicker_index(luma_means: List[float]) -> float:
    """
    Compute flicker index from luminance means.
    
    The flicker index is defined as the median of the frame-to-frame
    luminance deltas, representing the typical amount of brightness
    variation in the video.
    
    Args:
        luma_means: List of per-frame mean luminance values
    
    Returns:
        Flicker index value
    """
    if len(luma_means) < 2:
        return 0.0
    
    # Compute frame-to-frame deltas
    deltas = []
    for i in range(1, len(luma_means)):
        delta = abs(luma_means[i] - luma_means[i-1])
        deltas.append(delta)
    
    if not deltas:
        return 0.0
    
    # Return median delta as flicker index
    return float(np.median(deltas))


def compute_flicker_metrics(luma_means: List[float]) -> float:
    """
    Compute comprehensive flicker metrics.
    
    Args:
        luma_means: List of per-frame mean luminance values
    
    Returns:
        Primary flicker index metric
    """
    return compute_flicker_index(luma_means)


def compute_frame_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two frames.
    
    Args:
        frame1: First frame in float32 BGR format
        frame2: Second frame in float32 BGR format
    
    Returns:
        SSIM value between 0 and 1
    """
    # Convert to grayscale for SSIM computation
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    ssim_value = ssim(gray1, gray2, data_range=1.0)
    
    return float(ssim_value)


def analyze_luminance_distribution(luma_means: List[float]) -> Dict[str, float]:
    """
    Analyze the distribution of luminance values.
    
    Args:
        luma_means: List of per-frame mean luminance values
    
    Returns:
        Dictionary with distribution statistics
    """
    if not luma_means:
        return {}
    
    luma_array = np.array(luma_means)
    
    return {
        'mean_luminance': float(np.mean(luma_array)),
        'std_luminance': float(np.std(luma_array)),
        'min_luminance': float(np.min(luma_array)),
        'max_luminance': float(np.max(luma_array)),
        'luminance_range': float(np.max(luma_array) - np.min(luma_array)),
        'percentile_10': float(np.percentile(luma_array, 10)),
        'percentile_90': float(np.percentile(luma_array, 90)),
    }


def detect_flash_events(
    luma_means: List[float], 
    threshold: float = 0.12,
    min_duration: int = 1
) -> List[Tuple[int, int, float]]:
    """
    Detect flash events in the luminance sequence.
    
    Args:
        luma_means: List of per-frame mean luminance values
        threshold: Minimum delta to consider a flash
        min_duration: Minimum duration in frames for a flash event
    
    Returns:
        List of tuples (start_frame, end_frame, peak_delta)
    """
    if len(luma_means) < 3:
        return []
    
    flash_events = []
    in_flash = False
    flash_start = 0
    flash_peak = 0.0
    
    for i in range(1, len(luma_means) - 1):
        # Check for flash: current frame significantly brighter than neighbors
        delta_prev = luma_means[i] - luma_means[i-1]
        delta_next = luma_means[i] - luma_means[i+1]
        
        is_flash_frame = delta_prev > threshold and delta_next > threshold
        
        if is_flash_frame and not in_flash:
            # Start of flash event
            in_flash = True
            flash_start = i
            flash_peak = max(delta_prev, delta_next)
        
        elif is_flash_frame and in_flash:
            # Continue flash event
            flash_peak = max(flash_peak, max(delta_prev, delta_next))
        
        elif not is_flash_frame and in_flash:
            # End of flash event
            duration = i - flash_start
            if duration >= min_duration:
                flash_events.append((flash_start, i-1, flash_peak))
            
            in_flash = False
    
    # Handle flash event that extends to the end
    if in_flash:
        duration = len(luma_means) - 1 - flash_start
        if duration >= min_duration:
            flash_events.append((flash_start, len(luma_means) - 1, flash_peak))
    
    return flash_events


def compute_video_metrics(video_file: Path, sample_frames: int = 1000) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a video file.
    
    Args:
        video_file: Path to video file
        sample_frames: Maximum number of frames to sample for analysis
    
    Returns:
        Dictionary with all computed metrics
    """
    # Import locally to avoid circular import
    from destrobe.core.io import VideoCapture
    
    cap = VideoCapture(video_file)
    
    try:
        # Determine sampling strategy
        total_frames = cap.frame_count
        if total_frames <= sample_frames:
            # Process all frames
            frame_indices = list(range(total_frames))
        else:
            # Sample evenly across the video
            step = total_frames / sample_frames
            frame_indices = [int(i * step) for i in range(sample_frames)]
        
        # Extract frames and compute luminance
        luma_means = []
        frames_data = []
        
        with tqdm(total=len(frame_indices), desc="Analyzing video") as pbar:
            for frame_idx in frame_indices:
                cap.seek_to_frame(frame_idx)
                ret, frame = cap.read_frame()
                
                if not ret or frame is None:
                    break
                
                # Validate frame
                if not isinstance(frame, np.ndarray):
                    continue
                
                # Convert to float32
                frame_float = frame.astype(np.float32) / 255.0
                
                # Compute luminance
                gray = cv2.cvtColor(frame_float, cv2.COLOR_BGR2GRAY)
                luma_mean = np.mean(gray)
                luma_means.append(luma_mean)
                
                # Store frame data for additional analysis
                frames_data.append({
                    'frame_idx': frame_idx,
                    'luma_mean': luma_mean,
                    'luma_std': np.std(gray),
                    'luma_min': np.min(gray),
                    'luma_max': np.max(gray),
                })
                
                pbar.update(1)
        
        # Compute metrics
        metrics = {
            'video_file': str(video_file),
            'total_frames': total_frames,
            'sampled_frames': len(luma_means),
            'fps': cap.fps,
            'duration_seconds': cap.get_duration(),
            'resolution': f"{cap.width}x{cap.height}",
        }
        
        if luma_means:
            # Primary flicker metrics
            metrics['flicker_index'] = compute_flicker_index(luma_means)
            
            # Luminance distribution
            metrics.update(analyze_luminance_distribution(luma_means))
            
            # Flash detection
            flash_events = detect_flash_events(luma_means)
            metrics['flash_events_count'] = len(flash_events)
            
            if flash_events:
                peak_intensities = [event[2] for event in flash_events]
                metrics['max_flash_intensity'] = max(peak_intensities)
                metrics['avg_flash_intensity'] = np.mean(peak_intensities)
            else:
                metrics['max_flash_intensity'] = 0.0
                metrics['avg_flash_intensity'] = 0.0
            
            # Frame-to-frame statistics
            deltas = []
            for i in range(1, len(luma_means)):
                delta = abs(luma_means[i] - luma_means[i-1])
                deltas.append(delta)
            
            if deltas:
                metrics['mean_delta'] = np.mean(deltas)
                metrics['std_delta'] = np.std(deltas)
                metrics['max_delta'] = np.max(deltas)
                metrics['percentile_95_delta'] = np.percentile(deltas, 95)
            
            # Stability metrics
            metrics['luminance_stability'] = 1.0 / (1.0 + metrics.get('std_delta', 0.0))
        
        return metrics
    
    finally:
        cap.close()


def compare_videos(
    original_file: Path, 
    processed_file: Path, 
    sample_frames: int = 300
) -> Dict[str, Any]:
    """
    Compare metrics between original and processed videos.
    
    Args:
        original_file: Path to original video
        processed_file: Path to processed video
        sample_frames: Number of frames to sample for comparison
    
    Returns:
        Dictionary with comparison metrics
    """
    # Compute metrics for both videos
    original_metrics = compute_video_metrics(original_file, sample_frames)
    processed_metrics = compute_video_metrics(processed_file, sample_frames)
    
    # Compute relative improvements
    comparison = {
        'original_flicker_index': original_metrics.get('flicker_index', 0.0),
        'processed_flicker_index': processed_metrics.get('flicker_index', 0.0),
        'original_flash_events': original_metrics.get('flash_events_count', 0),
        'processed_flash_events': processed_metrics.get('flash_events_count', 0),
    }
    
    # Calculate improvement percentages
    if original_metrics.get('flicker_index', 0.0) > 0:
        flicker_reduction = (
            1.0 - processed_metrics.get('flicker_index', 0.0) / 
            original_metrics.get('flicker_index', 1.0)
        ) * 100
        comparison['flicker_reduction_percent'] = flicker_reduction
    
    if original_metrics.get('flash_events_count', 0) > 0:
        flash_reduction = (
            1.0 - processed_metrics.get('flash_events_count', 0) / 
            original_metrics.get('flash_events_count', 1)
        ) * 100
        comparison['flash_reduction_percent'] = flash_reduction
    
    return comparison
