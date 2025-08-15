"""
Video I/O, processing, and audio remux functionality for destrobe.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from destrobe.core.filters import apply_filter
from destrobe.core.enhanced_filters import enhanced_apply_filter
from destrobe.core.ultra_filters import ultra_apply_filter, hybrid_ultra_filter
from destrobe.core.extreme_filters import extreme_apply_filter
from destrobe.core.hyperextreme_filters import hyperextreme_apply_filter
from destrobe.core.metrics import compute_flicker_metrics, compute_frame_ssim
from destrobe.utils.fps import FPSCounter


class VideoCapture:
    """Enhanced video capture with better error handling."""
    
    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.cap = cv2.VideoCapture(str(filename))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {filename}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.fps <= 0 or self.frame_count <= 0:
            raise ValueError(f"Invalid video properties: fps={self.fps}, frames={self.frame_count}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame."""
        ret, frame = self.cap.read()
        return ret, frame
    
    def seek_to_time(self, seconds: float) -> bool:
        """Seek to a specific time in seconds."""
        frame_number = int(seconds * self.fps)
        return self.seek_to_frame(frame_number)
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """Seek to a specific frame number."""
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0
    
    def close(self) -> None:
        """Close the video capture."""
        if self.cap:
            self.cap.release()


class VideoWriter:
    """Enhanced video writer with codec fallbacks."""
    
    def __init__(
        self,
        filename: Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ) -> None:
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        
        # Try different codecs in order of preference
        codecs_to_try = [codec, "mp4v", "XVID", "MJPG"]
        
        self.writer = None
        for codec_name in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(
                str(filename), fourcc, fps, (width, height)
            )
            
            if writer.isOpened():
                self.writer = writer
                break
            else:
                writer.release()
        
        if self.writer is None:
            raise ValueError(f"Could not create video writer for {filename}")
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        if self.writer:
            self.writer.write(frame)
    
    def close(self) -> None:
        """Close the video writer."""
        if self.writer:
            self.writer.release()


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def remux_audio(video_file: Path, audio_source: Path, output_file: Path) -> bool:
    """
    Remux audio from source to video using FFmpeg.
    
    Args:
        video_file: Path to video file (no audio or to be replaced)
        audio_source: Path to source file with audio
        output_file: Path to output file with remuxed audio
    
    Returns:
        True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
    
    try:
        # FFmpeg command to remux audio
        cmd = [
            "ffmpeg",
            "-i", str(video_file),     # Video input
            "-i", str(audio_source),   # Audio input
            "-map", "0:v:0",           # Map video from first input
            "-map", "1:a:0?",          # Map audio from second input (optional)
            "-c", "copy",              # Copy streams without re-encoding
            "-y",                      # Overwrite output file
            str(output_file)
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return result.returncode == 0
        
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def parse_time_string(time_str: str) -> float:
    """Parse time string in HH:MM:SS format to seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS or MM:SS or SS")


class VideoProcessor:
    """Main video processing class that orchestrates the entire pipeline."""
    
    def __init__(
        self,
        method: str = "median3",
        strength: float = 0.5,
        flash_thresh: float = 0.12,
        no_audio: bool = False,
        threads: Optional[int] = None,
    ) -> None:
        self.method = method
        self.strength = strength
        self.flash_thresh = flash_thresh
        self.no_audio = no_audio
        
        # Set OpenCV thread count if specified
        if threads is not None:
            cv2.setNumThreads(threads)
    
    def process_video(
        self,
        input_file: Path,
        output_file: Path,
        progress_callback: Optional[Callable] = None,
        start_time: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file with the configured filter.
        
        Args:
            input_file: Input video file path
            output_file: Output video file path
            progress_callback: Optional callback for progress updates
            start_time: Optional start time in HH:MM:SS format
            duration: Optional duration in seconds
        
        Returns:
            Dictionary with processing metrics
        """
        
        # Open input video
        cap = VideoCapture(input_file)
        
        try:
            # Calculate processing range
            start_frame = 0
            if start_time:
                start_seconds = parse_time_string(start_time)
                start_frame = int(start_seconds * cap.fps)
            
            end_frame = cap.frame_count
            if duration:
                end_frame = min(end_frame, start_frame + int(duration * cap.fps))
            
            total_frames = end_frame - start_frame
            
            if total_frames <= 0:
                raise ValueError("No frames to process")
            
            # Seek to start position
            if start_frame > 0:
                cap.seek_to_frame(start_frame)
            
            # Create temporary output file for video-only
            temp_video = output_file.with_suffix('.temp.mp4')
            
            # Create video writer
            writer = VideoWriter(
                temp_video,
                cap.fps,
                cap.width,
                cap.height
            )
            
            try:
                # Initialize metrics tracking
                fps_counter = FPSCounter()
                fps_counter.start()
                
                original_frames = []
                processed_frames = []
                sample_size = min(300, total_frames)  # Sample for metrics
                sample_interval = max(1, total_frames // sample_size)
                
                # Process frames
                with tqdm(total=total_frames, desc="Processing", disable=progress_callback is not None) as pbar:
                    frame_buffer = []
                    processed_count = 0
                    
                    for i in range(total_frames):
                        ret, frame = cap.read_frame()
                        if not ret:
                            break
                        
                        # Convert to float32 for processing
                        frame_float = frame.astype(np.float32) / 255.0
                        
                        # Apply filter
                        if self.method in ["median3", "enhanced_median", "ultra_suppress", "ultra_stabilize", "hybrid_ultra", "extreme_smooth", "extreme_flash", "nuclear", "hyperextreme", "annihilation"]:
                            frame_buffer.append(frame_float)
                            
                            # Keep larger buffer for ultra/extreme/hyper methods
                            if self.method in ["hyperextreme", "annihilation"]:
                                max_buffer_size = 11
                            elif self.method.startswith("extreme_") or self.method == "nuclear":
                                max_buffer_size = 9
                            elif self.method.startswith("ultra_") or self.method == "hybrid_ultra":
                                max_buffer_size = 7
                            else:
                                max_buffer_size = 5
                            
                            if len(frame_buffer) > max_buffer_size:
                                frame_buffer.pop(0)
                            
                            # Process when we have enough frames
                            if self.method in ["hyperextreme", "annihilation"]:
                                min_frames = 9
                            elif self.method.startswith("extreme_") or self.method == "nuclear":
                                min_frames = 7
                            elif self.method.startswith("ultra_") or self.method == "hybrid_ultra":
                                min_frames = 5
                            else:
                                min_frames = 3
                            
                            if len(frame_buffer) >= min_frames:
                                if self.method in ["hyperextreme", "annihilation"]:
                                    processed_frame = hyperextreme_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                elif self.method.startswith("extreme_") or self.method == "nuclear":
                                    processed_frame = extreme_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                elif self.method.startswith("ultra_") or self.method == "hybrid_ultra":
                                    if self.method == "hybrid_ultra":
                                        processed_frame = hybrid_ultra_filter(
                                            frame_buffer, self.flash_thresh, self.strength
                                        )
                                    else:
                                        processed_frame = ultra_apply_filter(
                                            frame_buffer, self.method, self.strength, self.flash_thresh
                                        )
                                elif self.method.startswith("enhanced_"):
                                    processed_frame = enhanced_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                else:
                                    processed_frame = apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                
                                # Convert back to uint8 and write
                                output_frame = (processed_frame * 255).astype(np.uint8)
                                writer.write_frame(output_frame)
                                
                                # Collect samples for metrics
                                if processed_count % sample_interval == 0 and len(original_frames) < sample_size:
                                    original_frames.append(frame_buffer[1])  # Middle frame
                                    processed_frames.append(processed_frame)
                                
                                processed_count += 1
                                frame_buffer.pop(0)  # Remove oldest frame
                            
                            elif i == 0:
                                # First frame - write as-is
                                writer.write_frame(frame)
                                processed_count += 1
                        
                        else:
                            # For EMA and flashcap, we need different buffering
                            if self.method.startswith("enhanced_") or self.method.startswith("ultra_") or self.method.startswith("extreme_") or self.method in ["hyperextreme", "annihilation"]:
                                # Keep a buffer for enhanced/ultra/extreme/hyper methods
                                frame_buffer.append(frame_float)
                                
                                if self.method in ["hyperextreme", "annihilation"]:
                                    max_size = 11
                                elif self.method.startswith("extreme_"):
                                    max_size = 9
                                elif self.method.startswith("ultra_"):
                                    max_size = 7
                                else:
                                    max_size = 5
                                
                                if len(frame_buffer) > max_size:
                                    frame_buffer.pop(0)
                                
                                if self.method in ["hyperextreme", "annihilation"]:
                                    processed_frame = hyperextreme_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                elif self.method.startswith("extreme_"):
                                    processed_frame = extreme_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                elif self.method.startswith("ultra_"):
                                    processed_frame = ultra_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                                else:
                                    processed_frame = enhanced_apply_filter(
                                        frame_buffer, self.method, self.strength, self.flash_thresh
                                    )
                            else:
                                processed_frame = apply_filter(
                                    [frame_float], self.method, self.strength, self.flash_thresh
                                )
                            
                            output_frame = (processed_frame * 255).astype(np.uint8)
                            writer.write_frame(output_frame)
                            
                            # Collect samples for metrics
                            if processed_count % sample_interval == 0 and len(original_frames) < sample_size:
                                original_frames.append(frame_float)
                                processed_frames.append(processed_frame)
                            
                            processed_count += 1
                        
                        fps_counter.update()
                        pbar.update(1)
                        
                        if progress_callback:
                            progress_callback()
                    
                    # Handle remaining frames for median3
                    if self.method == "median3" and frame_buffer:
                        # Write remaining frames
                        for remaining_frame in frame_buffer:
                            output_frame = (remaining_frame * 255).astype(np.uint8)
                            writer.write_frame(output_frame)
                            processed_count += 1
                
            finally:
                writer.close()
            
            # Compute metrics
            metrics = {"frames_processed": processed_count}
            metrics.update(fps_counter.get_stats())
            
            if original_frames and processed_frames:
                # Compute flicker metrics
                original_lumas = [np.mean(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)) 
                                for f in original_frames]
                processed_lumas = [np.mean(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)) 
                                 for f in processed_frames]
                
                metrics["flicker_before"] = compute_flicker_metrics(original_lumas)
                metrics["flicker_after"] = compute_flicker_metrics(processed_lumas)
                
                # Compute SSIM on a subset
                ssim_scores = []
                for i in range(0, len(original_frames), max(1, len(original_frames) // 10)):
                    ssim = compute_frame_ssim(original_frames[i], processed_frames[i])
                    ssim_scores.append(ssim)
                
                metrics["ssim"] = np.mean(ssim_scores) if ssim_scores else 0.0
            
            # Handle audio remux
            final_output = output_file
            
            if not self.no_audio and check_ffmpeg():
                if remux_audio(temp_video, input_file, final_output):
                    # Remove temp file
                    temp_video.unlink()
                    metrics["audio_remuxed"] = True
                else:
                    # Fallback: rename temp file to final output
                    temp_video.rename(final_output)
                    metrics["audio_remuxed"] = False
            else:
                # No audio processing: rename temp file
                temp_video.rename(final_output)
                metrics["audio_remuxed"] = False
            
            return metrics
        
        finally:
            cap.close()
