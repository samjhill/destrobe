"""
Synthetic video generation for testing destrobe filters.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class SyntheticVideoGenerator:
    """Generate synthetic videos for testing flicker reduction algorithms."""
    
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        fps: float = 30.0,
        duration_seconds: float = 5.0
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.total_frames = int(fps * duration_seconds)
    
    def create_flash_video(
        self,
        output_path: Path,
        flash_interval: int = 10,
        flash_intensity: float = 0.9,
        base_brightness: float = 0.2
    ) -> None:
        """
        Create a video with periodic bright flashes.
        
        Args:
            output_path: Output video file path
            flash_interval: Frames between flashes
            flash_intensity: Brightness of flash frames (0-1)
            base_brightness: Brightness of normal frames (0-1)
        """
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        try:
            for frame_idx in range(self.total_frames):
                # Determine frame brightness
                if frame_idx % flash_interval == 0:
                    brightness = flash_intensity
                else:
                    brightness = base_brightness
                
                # Create frame
                frame = np.full((self.height, self.width, 3), brightness * 255, dtype=np.uint8)
                
                # Add some texture to make it more realistic
                noise = np.random.normal(0, 10, (self.height, self.width, 3))
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
                writer.write(frame)
        
        finally:
            writer.release()
    
    def create_strobe_video(
        self,
        output_path: Path,
        strobe_frequency: float = 10.0,
        bright_color: Tuple[int, int, int] = (255, 255, 255),
        dark_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        """
        Create a video with strobing pattern.
        
        Args:
            output_path: Output video file path
            strobe_frequency: Strobe frequency in Hz
            bright_color: RGB color for bright frames
            dark_color: RGB color for dark frames
        """
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        # Calculate frames per half-cycle
        frames_per_half_cycle = self.fps / (2 * strobe_frequency)
        
        try:
            for frame_idx in range(self.total_frames):
                # Determine if we're in bright or dark phase
                cycle_position = (frame_idx / frames_per_half_cycle) % 2
                
                if cycle_position < 1.0:
                    color = bright_color
                else:
                    color = dark_color
                
                # Create frame
                frame = np.full((self.height, self.width, 3), color, dtype=np.uint8)
                
                writer.write(frame)
        
        finally:
            writer.release()
    
    def create_gradual_brightness_video(
        self,
        output_path: Path,
        brightness_range: Tuple[float, float] = (0.1, 0.9),
        cycles: int = 3
    ) -> None:
        """
        Create a video with gradual brightness changes.
        
        Args:
            output_path: Output video file path
            brightness_range: Min and max brightness (0-1)
            cycles: Number of brightness cycles
        """
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        min_brightness, max_brightness = brightness_range
        brightness_amplitude = (max_brightness - min_brightness) / 2
        brightness_offset = min_brightness + brightness_amplitude
        
        try:
            for frame_idx in range(self.total_frames):
                # Calculate brightness using sine wave
                phase = 2 * np.pi * cycles * frame_idx / self.total_frames
                brightness = brightness_offset + brightness_amplitude * np.sin(phase)
                
                # Create frame
                frame_value = int(brightness * 255)
                frame = np.full((self.height, self.width, 3), frame_value, dtype=np.uint8)
                
                # Add gradient for visual interest
                gradient = np.linspace(0.8, 1.2, self.width)
                gradient = np.clip(gradient, 0.8, 1.2)
                
                for x in range(self.width):
                    frame[:, x] = np.clip(frame[:, x] * gradient[x], 0, 255)
                
                writer.write(frame)
        
        finally:
            writer.release()
    
    def create_random_noise_video(
        self,
        output_path: Path,
        base_brightness: float = 0.5,
        noise_amplitude: float = 0.3
    ) -> None:
        """
        Create a video with random brightness variations.
        
        Args:
            output_path: Output video file path
            base_brightness: Base brightness level (0-1)
            noise_amplitude: Amplitude of random variations (0-1)
        """
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        try:
            for frame_idx in range(self.total_frames):
                # Generate random brightness variation
                noise = np.random.normal(0, noise_amplitude)
                brightness = np.clip(base_brightness + noise, 0, 1)
                
                # Create frame
                frame_value = int(brightness * 255)
                frame = np.full((self.height, self.width, 3), frame_value, dtype=np.uint8)
                
                # Add spatial noise
                spatial_noise = np.random.normal(0, 15, (self.height, self.width, 3))
                frame = np.clip(frame.astype(np.float32) + spatial_noise, 0, 255).astype(np.uint8)
                
                writer.write(frame)
        
        finally:
            writer.release()
    
    def create_scene_cut_video(
        self,
        output_path: Path,
        scene_duration_frames: int = 30,
        brightness_levels: List[float] = None
    ) -> None:
        """
        Create a video with abrupt scene cuts (hard cuts).
        
        Args:
            output_path: Output video file path
            scene_duration_frames: Frames per scene
            brightness_levels: List of brightness levels for different scenes
        """
        
        if brightness_levels is None:
            brightness_levels = [0.2, 0.8, 0.4, 0.9, 0.1, 0.6]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        try:
            for frame_idx in range(self.total_frames):
                # Determine current scene
                scene_idx = (frame_idx // scene_duration_frames) % len(brightness_levels)
                brightness = brightness_levels[scene_idx]
                
                # Create frame with different colors for each scene
                hue = (scene_idx * 60) % 180  # Different hues for visual distinction
                
                # Create HSV frame and convert to BGR
                hsv_frame = np.full((self.height, self.width, 3), 
                                   [hue, 128, int(brightness * 255)], dtype=np.uint8)
                frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
                
                writer.write(frame)
        
        finally:
            writer.release()
    
    def create_animation_video(
        self,
        output_path: Path,
        moving_object_size: int = 20,
        background_brightness: float = 0.1,
        object_brightness: float = 0.9
    ) -> None:
        """
        Create a video with animated content (moving object).
        
        Args:
            output_path: Output video file path
            moving_object_size: Size of moving object in pixels
            background_brightness: Brightness of background (0-1)
            object_brightness: Brightness of moving object (0-1)
        """
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        try:
            for frame_idx in range(self.total_frames):
                # Create background
                bg_value = int(background_brightness * 255)
                frame = np.full((self.height, self.width, 3), bg_value, dtype=np.uint8)
                
                # Calculate object position (circular motion)
                center_x = self.width // 2
                center_y = self.height // 2
                radius = min(center_x, center_y) - moving_object_size
                
                angle = 2 * np.pi * frame_idx / (self.fps * 2)  # 2-second orbit
                obj_x = int(center_x + radius * np.cos(angle))
                obj_y = int(center_y + radius * np.sin(angle))
                
                # Draw moving object
                obj_value = int(object_brightness * 255)
                cv2.circle(frame, (obj_x, obj_y), moving_object_size, 
                          (obj_value, obj_value, obj_value), -1)
                
                writer.write(frame)
        
        finally:
            writer.release()


def create_test_video_suite(output_dir: Path) -> List[Path]:
    """
    Create a complete suite of test videos.
    
    Args:
        output_dir: Directory to save test videos
    
    Returns:
        List of created video file paths
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticVideoGenerator(duration_seconds=3.0)  # Short videos for testing
    
    video_files = []
    
    # Flash video - periodic bright flashes
    flash_video = output_dir / "flash_test.mp4"
    generator.create_flash_video(flash_video, flash_interval=15)
    video_files.append(flash_video)
    
    # Strobe video - rapid alternating
    strobe_video = output_dir / "strobe_test.mp4"
    generator.create_strobe_video(strobe_video, strobe_frequency=8.0)
    video_files.append(strobe_video)
    
    # Gradual brightness changes
    gradual_video = output_dir / "gradual_test.mp4"
    generator.create_gradual_brightness_video(gradual_video, cycles=2)
    video_files.append(gradual_video)
    
    # Random noise
    noise_video = output_dir / "noise_test.mp4"
    generator.create_random_noise_video(noise_video, noise_amplitude=0.2)
    video_files.append(noise_video)
    
    # Scene cuts
    cuts_video = output_dir / "cuts_test.mp4"
    generator.create_scene_cut_video(cuts_video, scene_duration_frames=20)
    video_files.append(cuts_video)
    
    # Animation
    animation_video = output_dir / "animation_test.mp4"
    generator.create_animation_video(animation_video)
    video_files.append(animation_video)
    
    return video_files
