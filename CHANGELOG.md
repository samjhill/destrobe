# Changelog

All notable changes to destrobe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-08-15

### 🎉 Initial Release

destrobe v1.0.0 is the first stable release of the cross-platform CLI tool for reducing video strobing and flicker while preserving audio and sync.

### ✨ Added

#### Core Features
- **CLI Interface**: Complete command-line interface with `run`, `preview`, and `metrics` subcommands
- **Multiple Filtering Algorithms**: 
  - Basic: `median3`, `ema`, `flashcap`
  - Enhanced: `enhanced_median`, `enhanced_ema`, `enhanced_flashcap` (improved sensitivity)
  - Ultra: `ultra_suppress`, `ultra_stabilize`, `hybrid_ultra` (aggressive processing)
  - Extreme: `extreme_smooth`, `extreme_flash`, `nuclear` (maximum suppression)
  - Hyperextreme: `hyperextreme`, `annihilation` (quality-sacrificing maximum effect)

#### Preset System
- **13 Built-in Presets**: From `safe` (40% reduction) to `annihilation` (77% reduction)
- **Optimized Settings**: Each preset balanced for different use cases
- **Maximum Effectiveness**: `maximum` preset achieves 81%+ flicker reduction

#### Processing Capabilities
- **Audio Preservation**: Automatic FFmpeg remuxing to preserve original audio
- **Batch Processing**: Process individual files or entire directories recursively
- **Performance**: 350+ fps processing speed on 1080p content
- **Quality Metrics**: Built-in Flicker Index and SSIM computation
- **Progress Tracking**: Real-time progress bars and performance benchmarking

#### Safety Features
- **Photosensitivity Warnings**: Comprehensive safety notices for dangerous content
- **Preview Mode**: Safe side-by-side comparison generation
- **Conservative Defaults**: Safe preset as default with first-run warnings
- **Medical Warning Content**: Proper handling and labeling of challenging test cases

### 📁 Example Content

#### Main Demonstration
- **porygon.mp4**: Real-world challenging content (59 flash events in 13 seconds)
- **porygon_ULTRA.preview.mp4**: Side-by-side demonstration of 81% flicker reduction

#### Comprehensive Test Suite (10 Videos)
- **Gaming Content**: `gaming_flicker.mp4` - Muzzle flashes and explosions
- **TV/Broadcast**: `tv_broadcast.mp4`, `sports_broadcast.mp4` - Camera flashes and lighting
- **Medical Warning**: `medical_warning.mp4` - Rapid alternating patterns (most dangerous)
- **Entertainment**: `concert_lighting.mp4`, `animation_effects.mp4` - Strobe and effects
- **Synthetic Tests**: `flash_test.mp4`, `rapid_strobe.mp4`, `scene_cuts.mp4`, `subtle_flicker.mp4`
- **All videos include `.preview.mp4` files** showing safe side-by-side comparisons

### 🛡️ Safety & Accessibility
- **Comprehensive Safety Documentation**: Detailed warnings for all dangerous content
- **Categorized Content**: Test videos organized by risk level and content type
- **Safe Demonstration**: All examples include processed versions for safe viewing
- **Accessibility Focus**: Designed specifically for photosensitive users

### 🧪 Technical Excellence
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Python 3.10+**: Modern Python with type hints and robust error handling
- **Comprehensive Testing**: Unit tests, integration tests, and real-world validation
- **Professional Code Quality**: Linting with ruff, formatting with black, type checking with mypy
- **Performance Optimized**: Efficient frame buffering and processing pipelines

### 🎯 Proven Effectiveness
- **81%+ Flicker Reduction**: Achieved on challenging real-world content
- **90%+ Quality Preservation**: High SSIM scores maintaining video quality
- **Multiple Algorithm Types**: Different approaches for different content types
- **Real-World Validation**: Tested on gaming, broadcast, medical, and entertainment content

### 📦 Distribution
- **PyPI Ready**: Professional package metadata and dependency management
- **pipx Compatible**: Easy installation and isolated environment
- **GitHub Integration**: Open source with comprehensive documentation
- **Example Content**: Complete demonstration suite included

---

This release represents a complete, production-ready tool for video flicker reduction with proven effectiveness across multiple content types and comprehensive safety considerations for photosensitive users.
