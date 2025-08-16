"""
Video input normalization for destrobe.

Converts various video formats (.webm, .mkv, .avi, etc.) to MP4 with 
compatible codecs before processing to ensure consistent handling.
"""

import json
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console

console = Console()


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None


def probe_video_format(video_file: Path) -> Optional[dict]:
    """
    Probe video file to get format information using ffprobe.
    
    Args:
        video_file: Path to the video file
        
    Returns:
        Dictionary with video format info, or None if probe fails
    """
    if not check_ffmpeg():
        return None
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_format",
            "-show_streams",
            "-select_streams", "v:0",
            "-print_format", "json",
            str(video_file)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data
        
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    
    return None


def needs_normalization(video_file: Path) -> bool:
    """
    Check if a video file needs normalization based on format and codec.
    
    Args:
        video_file: Path to the video file
        
    Returns:
        True if normalization is needed, False otherwise
    """
    # Check file extension first
    extension = video_file.suffix.lower()
    
    # These extensions typically need normalization
    if extension in ['.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv', '.3gp', '.ogg']:
        return True
    
    # MP4 files might still need normalization if they use unsupported codecs
    if extension == '.mp4':
        format_info = probe_video_format(video_file)
        if format_info and 'streams' in format_info:
            for stream in format_info['streams']:
                if stream.get('codec_type') == 'video':
                    codec = stream.get('codec_name', '').lower()
                    # Check for problematic codecs
                    if codec in ['vp8', 'vp9', 'av1', 'hevc', 'h265']:
                        return True
        
        # If probe failed, assume MP4 is fine
        return False
    
    return False


def normalize_video(
    input_file: Path, 
    output_file: Optional[Path] = None,
    preserve_quality: bool = True,
    progress_callback: Optional[callable] = None
) -> Tuple[bool, Optional[Path]]:
    """
    Normalize a video file to MP4 with H.264 codec.
    
    Args:
        input_file: Path to input video file
        output_file: Optional output path. If None, creates temp file
        preserve_quality: If True, use higher quality settings
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (success: bool, output_path: Optional[Path])
    """
    if not check_ffmpeg():
        console.print("[yellow]Warning:[/yellow] FFmpeg not found, skipping normalization")
        return False, None
    
    # Create output path if not provided
    if output_file is None:
        temp_dir = tempfile.gettempdir()
        output_file = Path(temp_dir) / f"{input_file.stem}_normalized.mp4"
    
    try:
        # Base FFmpeg command for normalization
        cmd = [
            "ffmpeg",
            "-i", str(input_file),
            "-c:v", "libx264",  # Use H.264 codec
            "-c:a", "aac",      # Use AAC for audio
            "-movflags", "+faststart",  # Optimize for streaming
            "-y",  # Overwrite output file
        ]
        
        # Quality settings
        if preserve_quality:
            cmd.extend([
                "-crf", "18",  # High quality
                "-preset", "medium",  # Balance speed vs compression
                "-profile:v", "high",  # H.264 profile
                "-level", "4.1",  # H.264 level
            ])
        else:
            cmd.extend([
                "-crf", "23",  # Standard quality
                "-preset", "fast",  # Faster encoding
                "-profile:v", "main",  # Compatibility profile
            ])
        
        # Add output file
        cmd.append(str(output_file))
        
        # Run FFmpeg
        if progress_callback:
            progress_callback()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        success = result.returncode == 0
        
        if success and output_file.exists():
            return True, output_file
        else:
            # Log error if available
            if result.stderr:
                console.print(f"[yellow]FFmpeg error:[/yellow] {result.stderr[:200]}...")
            return False, None
            
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        console.print(f"[yellow]Normalization failed:[/yellow] {str(e)}")
        return False, None


def get_normalized_input(
    input_file: Path,
    temp_dir: Optional[Path] = None,
    preserve_quality: bool = True,
    progress_callback: Optional[callable] = None
) -> Tuple[Path, bool]:
    """
    Get a normalized version of the input file, creating one if needed.
    
    Args:
        input_file: Original input file
        temp_dir: Directory for temporary files
        preserve_quality: If True, use higher quality normalization
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (file_path: Path, was_normalized: bool)
        - file_path: Path to use for processing (original or normalized)
        - was_normalized: True if a new normalized file was created
    """
    if not needs_normalization(input_file):
        return input_file, False
    
    console.print(f"[cyan]Normalizing input:[/cyan] {input_file.name}")
    console.print(f"[dim]This may take several minutes for large files...[/dim]")
    
    # Create temp file path
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir()) / "destrobe_normalized"
        temp_dir.mkdir(exist_ok=True)
    
    normalized_file = temp_dir / f"{input_file.stem}_normalized.mp4"
    
    # Normalize the video
    success, output_path = normalize_video(
        input_file,
        normalized_file,
        preserve_quality=preserve_quality,
        progress_callback=progress_callback
    )
    
    if success and output_path:
        console.print(f"[green]✓[/green] Normalized: {output_path.name}")
        return output_path, True
    else:
        console.print(f"[yellow]Warning:[/yellow] Normalization failed, using original file")
        return input_file, False


def cleanup_normalized_file(file_path: Path, was_normalized: bool) -> None:
    """
    Clean up a normalized file if it was created during processing.
    
    Args:
        file_path: Path to the file
        was_normalized: Whether the file was created by normalization
    """
    if was_normalized and file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass  # Ignore cleanup errors


# Add comprehensive format support
SUPPORTED_EXTENSIONS = [
    # Already supported
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
    # Additional formats that can be normalized
    ".flv", ".wmv", ".3gp", ".ogg", ".ogv", ".mpg", ".mpeg", 
    ".mp2", ".mpe", ".mpv", ".m2v", ".m4p", ".m4b", ".m4r",
    ".ts", ".mts", ".m2ts", ".vob", ".asf", ".rm", ".rmvb",
    ".dv", ".f4v", ".f4p", ".f4a", ".f4b"
]
