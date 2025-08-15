"""
CLI interface for destrobe - video strobing reduction tool.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from destrobe.core.io import VideoProcessor
from destrobe.core.metrics import compute_video_metrics
from destrobe.core.preview import create_preview
from destrobe.utils.logging import setup_logging
from destrobe.utils.fps import benchmark_system

app = typer.Typer(
    name="destrobe",
    help="CLI tool for reducing video strobing and flicker",
    add_completion=False,
)
console = Console()

# Presets mapping
PRESETS = {
    "safe": {"method": "flashcap", "strength": 0.7, "flash_thresh": 0.10},
    "balanced": {"method": "median3", "strength": 0.5, "flash_thresh": 0.12},
    "strong": {"method": "ema", "strength": 0.75, "flash_thresh": 0.12},
    "enhanced_safe": {"method": "enhanced_flashcap", "strength": 0.8, "flash_thresh": 0.04},
    "enhanced_balanced": {"method": "enhanced_median", "strength": 0.6, "flash_thresh": 0.05},
    "enhanced_strong": {"method": "enhanced_ema", "strength": 0.85, "flash_thresh": 0.03},
    "ultra_safe": {"method": "enhanced_flashcap", "strength": 0.95, "flash_thresh": 0.01},
    "ultra_strong": {"method": "enhanced_median", "strength": 0.9, "flash_thresh": 0.005},
    "maximum": {"method": "enhanced_flashcap", "strength": 0.99, "flash_thresh": 0.001},
    "extreme": {"method": "extreme_flash", "strength": 0.98, "flash_thresh": 0.002},
    "nuclear": {"method": "nuclear", "strength": 0.99, "flash_thresh": 0.001},
    "hyperextreme": {"method": "hyperextreme", "strength": 0.995, "flash_thresh": 0.0005},
    "annihilation": {"method": "annihilation", "strength": 0.999, "flash_thresh": 0.0001},
}

CONFIG_DIR = Path.home() / ".config" / "destrobe"
FIRST_RUN_FILE = CONFIG_DIR / ".first_run"


def check_first_run(no_warn: bool = False) -> None:
    """Check if this is the first run and show safety warning."""
    if no_warn:
        return
    
    if not FIRST_RUN_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        console.print(
            "[yellow]First-run notice:[/yellow] Reducing flashes, but cannot guarantee "
            "complete removal. Preview first if you are photosensitive. Use at your "
            "discretion. (hide with --no-warn)"
        )
        FIRST_RUN_FILE.touch()


def discover_video_files(
    inputs: List[Path], recursive: bool = False, extensions: List[str] = None
) -> List[Path]:
    """Discover video files from input paths."""
    if extensions is None:
        extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]
    
    video_files = []
    
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() in extensions:
                video_files.append(input_path)
        elif input_path.is_dir():
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for ext in extensions:
                video_files.extend(input_path.glob(f"{pattern}{ext}"))
                video_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(set(video_files))


@app.command()
def run(
    inputs: List[Path] = typer.Argument(..., help="Input video files or directories"),
    method: str = typer.Option("median3", "-m", "--method", help="Processing method"),
    strength: float = typer.Option(0.5, "-s", "--strength", help="Filter strength (0.0-1.0)"),
    flash_thresh: float = typer.Option(0.12, "--flash-thresh", help="Flash detection threshold"),
    outdir: Path = typer.Option(Path("destrobed"), "-o", "--outdir", help="Output directory"),
    recursive: bool = typer.Option(False, "--recursive", help="Process directories recursively"),
    ext: str = typer.Option(".mp4", "--ext", help="Output file extension"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio remux"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
    logfile: Optional[Path] = typer.Option(None, "--logfile", help="JSONL log file path"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Print performance metrics"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Use preset configuration"),
    threads: Optional[int] = typer.Option(None, "--threads", help="Number of processing threads"),
    no_warn: bool = typer.Option(False, "--no-warn", help="Skip photosensitivity warning"),
) -> None:
    """Process video files to reduce strobing and flicker."""
    
    check_first_run(no_warn)
    
    # Apply preset if specified
    if preset:
        if preset not in PRESETS:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
            raise typer.Exit(1)
        
        preset_config = PRESETS[preset]
        method = preset_config["method"]
        strength = preset_config["strength"]
        flash_thresh = preset_config["flash_thresh"]
    
    # Validate parameters
    valid_methods = [
        "median3", "ema", "flashcap", 
        "enhanced_median", "enhanced_ema", "enhanced_flashcap",
        "ultra_suppress", "ultra_stabilize", "hybrid_ultra",
        "extreme_smooth", "extreme_flash", "nuclear",
        "hyperextreme", "annihilation"
    ]
    if method not in valid_methods:
        console.print(f"[red]Error:[/red] Unknown method '{method}'. Available: {', '.join(valid_methods)}")
        raise typer.Exit(1)
    
    if not 0.0 <= strength <= 1.0:
        console.print("[red]Error:[/red] Strength must be between 0.0 and 1.0")
        raise typer.Exit(1)
    
    # Setup logging
    logger = setup_logging(logfile)
    
    # Discover video files
    video_files = discover_video_files(inputs, recursive)
    
    if not video_files:
        console.print("[yellow]Warning:[/yellow] No video files found")
        return
    
    console.print(f"Found {len(video_files)} video file(s)")
    
    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Setup processor
    processor = VideoProcessor(
        method=method,
        strength=strength,
        flash_thresh=flash_thresh,
        no_audio=no_audio,
        threads=threads,
    )
    
    # Process files
    with Progress() as progress:
        task = progress.add_task("Processing videos...", total=len(video_files))
        
        for video_file in video_files:
            progress.update(task, description=f"Processing {video_file.name}")
            
            # Generate output filename
            stem = video_file.stem
            suffix = f".{method}" if method != "median3" else ".median3"
            output_file = outdir / f"{stem}{suffix}{ext}"
            
            if output_file.exists() and not overwrite:
                console.print(f"[yellow]Skipping:[/yellow] {output_file} already exists")
                progress.advance(task)
                continue
            
            try:
                # Process the video
                metrics = processor.process_video(
                    video_file, output_file, progress_callback=lambda: progress.advance(task)
                )
                
                # Log metrics
                if logger:
                    log_entry = {
                        "file": str(video_file),
                        "output": str(output_file),
                        "method": method,
                        "strength": strength,
                        "flash_thresh": flash_thresh,
                        **metrics,
                    }
                    logger.info(log_entry)
                
                # Print summary
                if "flicker_before" in metrics and "flicker_after" in metrics:
                    reduction = (1 - metrics["flicker_after"] / metrics["flicker_before"]) * 100
                    console.print(
                        f"→ {video_file.name} → {output_file.name}\n"
                        f"  FI: {metrics['flicker_before']:.3f} → {metrics['flicker_after']:.3f} "
                        f"({reduction:+.1f}%), SSIM: {metrics.get('ssim', 0):.3f}"
                    )
                
                if benchmark and "fps" in metrics:
                    console.print(f"  Performance: {metrics['fps']:.1f} fps")
                
            except Exception as e:
                console.print(f"[red]Error processing {video_file.name}:[/red] {e}")
                
            progress.advance(task)
    
    console.print("[green]✓[/green] Processing complete")


@app.command()
def preview(
    input_file: Path = typer.Argument(..., help="Input video file"),
    method: str = typer.Option("median3", "-m", "--method", help="Processing method"),
    strength: float = typer.Option(0.5, "-s", "--strength", help="Filter strength (0.0-1.0)"),
    flash_thresh: float = typer.Option(0.12, "--flash-thresh", help="Flash detection threshold"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Use preset configuration"),
    seconds: int = typer.Option(10, "--seconds", help="Preview duration in seconds"),
    start: str = typer.Option("00:00:30", "--start", help="Start time (HH:MM:SS)"),
) -> None:
    """Create a side-by-side preview of the processing effect."""
    
    # Apply preset if specified
    if preset:
        if preset not in PRESETS:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
            raise typer.Exit(1)
        
        preset_config = PRESETS[preset]
        method = preset_config["method"]
        strength = preset_config["strength"]
        flash_thresh = preset_config["flash_thresh"]
    
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)
    
    output_file = input_file.parent / f"{input_file.stem}.preview.mp4"
    
    console.print(f"Creating preview: {input_file.name} → {output_file.name}")
    
    try:
        create_preview(
            input_file=input_file,
            output_file=output_file,
            method=method,
            strength=strength,
            flash_thresh=flash_thresh,
            duration_seconds=seconds,
            start_time=start,
        )
        console.print(f"[green]✓[/green] Preview created: {output_file}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def metrics(
    input_file: Path = typer.Argument(..., help="Input video file"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Compute and display flicker metrics for a video file."""
    
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)
    
    console.print(f"Computing metrics for: {input_file.name}")
    
    try:
        metrics_data = compute_video_metrics(input_file)
        
        if json_output:
            import json
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            metrics_json = convert_numpy_types(metrics_data)
            print(json.dumps(metrics_json, indent=2))
        else:
            console.print("\n[bold]Video Metrics:[/bold]")
            for key, value in metrics_data.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.4f}")
                else:
                    console.print(f"  {key}: {value}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.callback()
def main() -> None:
    """destrobe - CLI tool for reducing video strobing and flicker."""
    pass


if __name__ == "__main__":
    app()
