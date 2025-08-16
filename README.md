# destrobe

**CLI tool for reducing video strobing and flicker**

A cross-platform Python CLI that reduces strobing/flicker in videos while preserving audio and sync. Designed to be simple, fast, and safe-by-default for photosensitivity concerns.

## ⚠️ Important Safety Notice

**destrobe reduces flashing but cannot guarantee complete removal of all photosensitive triggers.** Always preview content first if you have photosensitive epilepsy or similar conditions. Use at your own discretion.

### ⚠️ Example Files Warning

**The example videos in the `examples/` directory contain rapid flashing and strobing effects that may trigger photosensitive epilepsy or seizures.** These files are included specifically to demonstrate destrobe's flicker reduction capabilities on challenging content. **DO NOT VIEW the original files directly if you have photosensitive epilepsy or related conditions.** Only view the processed versions (e.g., `examples/porygon_ULTRA.preview.mp4`) which have had the flashing significantly reduced.

## Quick Start

```bash
# Install with pipx (recommended)
pipx install destrobe

# Basic usage - process a video with default settings
destrobe run input.mp4

# Preview before processing (recommended)
destrobe preview input.mp4

# Batch process a directory
destrobe run /path/to/videos --recursive

# Use stronger filtering
destrobe run input.mp4 --preset strong
```

## Features

- **Three filtering methods**: Temporal median (`median3`), motion-aware smoothing (`ema`), and flash detection/capping (`flashcap`)
- **Safe presets**: `safe`, `balanced`, and `strong` configurations for different needs
- **Audio preservation**: Automatically remux original audio using FFmpeg when available
- **Batch processing**: Process individual files or entire directories recursively
- **Preview mode**: Generate side-by-side comparisons before full processing
- **Quality metrics**: Measure flicker reduction and structural similarity
- **Performance monitoring**: Built-in benchmarking and progress tracking
- **Cross-platform**: Works on macOS, Linux, and Windows

## Installation

### Using pipx (Recommended)

```bash
# Install from PyPI (simplest)
pipx install destrobe

# Or install from GitHub (latest development version)
pipx install git+https://github.com/samjhill/destrobe.git
```

### Using pip

```bash
# Install from PyPI
pip install destrobe

# Or install from GitHub
pip install git+https://github.com/samjhill/destrobe.git
```

### From Source

```bash
git clone https://github.com/samhilll/destrobe.git
cd destrobe
pip install -e .
```

### Requirements

- Python 3.10+
- FFmpeg (optional, for audio remux)
- Dependencies: OpenCV, NumPy, scikit-image, typer, rich, tqdm

## Usage

### Basic Commands

#### Process Videos

```bash
# Process single file
destrobe run video.mp4

# Process multiple files
destrobe run video1.mp4 video2.mkv video3.avi

# Process directory (non-recursive)
destrobe run /path/to/videos

# Process directory recursively
destrobe run /path/to/videos --recursive
```

#### Preview Mode

```bash
# Create 10-second preview starting at 30 seconds
destrobe preview video.mp4

# Custom preview settings
destrobe preview video.mp4 --seconds 15 --start 00:01:30 --method ema
```

#### Analyze Metrics

```bash
# Show flicker metrics
destrobe metrics video.mp4

# Output as JSON
destrobe metrics video.mp4 --json
```

### Filtering Methods

#### `median3` (Default)
Temporal median filter using 3-frame window. Excellent for removing isolated flashes while preserving motion.

```bash
destrobe run video.mp4 --method median3
```

#### `ema` (Exponential Moving Average)
Motion-aware temporal smoothing. Adapts filtering strength based on detected motion.

```bash
destrobe run video.mp4 --method ema --strength 0.7
```

#### `flashcap` (Flash Detection & Capping)
Detects sudden brightness spikes and caps them. Best for content with known flash patterns.

```bash
destrobe run video.mp4 --method flashcap --flash-thresh 0.10
```

### Presets

#### `safe` (Recommended for sensitive viewers)
- Method: `flashcap`
- Strength: 0.7
- Flash threshold: 0.10

```bash
destrobe run video.mp4 --preset safe
```

#### `balanced` (Default)
- Method: `median3`
- Balanced between quality and flicker reduction

```bash
destrobe run video.mp4 --preset balanced
```

#### `strong` (Maximum reduction)
- Method: `ema`
- Strength: 0.75
- More aggressive filtering

```bash
destrobe run video.mp4 --preset strong
```

### Advanced Options

```bash
destrobe run input.mp4 \
  --method median3 \
  --strength 0.6 \
  --flash-thresh 0.12 \
  --outdir processed \
  --ext .mp4 \
  --logfile metrics.jsonl \
  --benchmark \
  --threads 4 \
  --overwrite
```

#### Output Control
- `--outdir`: Output directory (default: `destrobed`)
- `--ext`: Output file extension (default: `.mp4`)
- `--overwrite`: Allow overwriting existing files
- `--no-audio`: Skip audio remux (video only)

#### Performance & Logging
- `--benchmark`: Show processing speed and system info
- `--logfile`: Save detailed metrics to JSONL file
- `--threads`: Number of processing threads

#### Other Options
- `--no-warn`: Skip photosensitivity warning
- `--recursive`: Process directories recursively

## Output

### File Naming
By default, processed files are saved with method suffix:
- `input.mp4` → `destrobed/input.median3.mp4`
- `input.mp4` → `destrobed/input.ema.mp4` (when using EMA)
- `input.mp4` → `destrobed/input.flashcap.mp4` (when using flashcap)

### Console Output
```
→ SailorMoon_EP01.mp4 → destrobed/SailorMoon_EP01.median3.mp4
  FI: 0.082 → 0.041 (-50.0%), SSIM: 0.967
  Performance: 58.2 fps
```

### Metrics Logging
When using `--logfile`, detailed metrics are saved as JSON Lines:

```json
{
  "file": "input.mp4",
  "output": "destrobed/input.median3.mp4", 
  "method": "median3",
  "flicker_before": 0.082,
  "flicker_after": 0.041,
  "ssim": 0.967,
  "fps": 58.2,
  "duration_s": 132.4,
  "audio_remuxed": true
}
```

## Performance

Typical performance on modern hardware:

| Resolution | Method | Apple M1 | Intel i7 | Notes |
|------------|--------|----------|----------|-------|
| 1080p | median3 | ~80 fps | ~60 fps | Real-time+ |
| 1080p | ema | ~90 fps | ~70 fps | Fastest |
| 1080p | flashcap | ~75 fps | ~55 fps | Most thorough |
| 4K | median3 | ~25 fps | ~18 fps | Usable |

Performance can be improved by:
- Using `--threads` to match your CPU cores
- Processing shorter segments for very large files
- Using `ema` method for fastest processing

## Technical Details

### Algorithms

#### Temporal Median Filter (`median3`)
- Uses 3-frame sliding window
- Computes per-pixel median on luminance channel
- Preserves chroma from center frame
- Excellent for isolated flashes, minimal motion blur

#### Exponential Moving Average (`ema`)
- Motion-aware temporal smoothing
- Adapts alpha based on inter-frame motion
- Higher motion = less smoothing (preserves action scenes)
- Good balance of speed and quality

#### Flash Detection & Capping (`flashcap`)  
- Detects sudden luminance spikes
- Caps detected flashes by blending with neighbors
- Applies mild temporal smoothing to other frames
- Best for known problematic content

### Quality Metrics

#### Flicker Index (FI)
Median of frame-to-frame luminance deltas. Lower values indicate less flicker.

#### Structural Similarity (SSIM)
Measures how well the processed video preserves the original's structure. Values near 1.0 indicate high quality preservation.

### Audio Handling
- Original audio is preserved via FFmpeg remux when available
- Falls back to video-only output if FFmpeg is missing
- No re-encoding of audio streams (fast and lossless)

## Troubleshooting

### FFmpeg Not Found
```bash
# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Slow Processing
- Use `--threads N` where N is your CPU core count
- Try `ema` method for fastest processing
- Process shorter clips first to test settings
- Consider reducing input resolution for very large files

### Poor Quality Results
- Try different methods: `median3` for flashes, `ema` for general smoothing
- Adjust `--strength` (lower = less filtering)
- Use `--preview` to test settings before full processing
- Check `--flash-thresh` for flashcap method

### Large File Sizes
- Output uses same codec as input when possible
- Use `--ext .mp4` for better compression
- Original audio is copied without re-encoding

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=destrobe

# Run linting
ruff check .
black --check .
mypy destrobe/
```

### Creating Test Videos

```python
from destrobe.tests.synthetic_video import create_test_video_suite
from pathlib import Path

# Generate test videos
test_videos = create_test_video_suite(Path("test_outputs"))
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and linting is clean
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with OpenCV for video processing
- Uses scikit-image for SSIM computation
- CLI powered by Typer and Rich
- Inspired by the need for accessible video content

## Related Projects

- [VapourSynth](http://www.vapoursynth.com/) - Advanced video processing framework
- [FFmpeg](https://ffmpeg.org/) - Multimedia framework used for audio remux
- [OpenCV](https://opencv.org/) - Computer vision library for video I/O

---

**Remember**: This tool reduces flashes but cannot guarantee complete removal. Always preview content if you have photosensitive conditions.
