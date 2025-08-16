# Destrobe Examples

This directory contains example videos demonstrating destrobe's flicker reduction capabilities.

## ⚠️ Safety Warning

**The videos in this directory contain rapid flashing and strobing effects that may trigger photosensitive epilepsy or seizures.** These files are included specifically to demonstrate destrobe's effectiveness on challenging content. 

**DO NOT VIEW the original files directly if you have photosensitive epilepsy or related conditions.** Only view the processed versions which have had the flashing significantly reduced.

## Example Files

### Main Examples

- **`porygon.mp4`** - Real-world video with intense rapid flashing (59 flash events in 13 seconds)
  - ⚠️ **DANGEROUS**: Contains rapid strobing effects
  - Original Flicker Index: 8.577 (very high)
  
- **`porygon_ULTRA.preview.mp4`** - Side-by-side comparison showing 81% flicker reduction
  - ✅ **SAFE**: Processed version with dramatically reduced flashing
  - Shows original vs processed with maximum preset
  - Demonstrates destrobe's effectiveness

### Test Videos

Located in `test_videos/` - **All videos have corresponding `.preview.mp4` files showing side-by-side comparisons:**

#### 🎮 **Gaming Content**
- **`gaming_flicker.mp4`** - ⚠️ Gaming scenario with muzzle flashes, explosions, and screen effects
  - *Preview*: `gaming_flicker.preview.mp4` - Shows enhanced_flashcap reducing gaming flashes
  - *Best method*: enhanced_flashcap (98% strength, 0.002 threshold)

#### 📺 **TV/Broadcast Content**  
- **`tv_broadcast.mp4`** - ⚠️ TV broadcast with camera flashes and bright scene transitions
  - *Preview*: `tv_broadcast.preview.mp4` - Shows flash reduction in broadcast content
  - *Best method*: enhanced_flashcap (95% strength, 0.005 threshold)

- **`sports_broadcast.mp4`** - ⚠️ Sports event with stadium lighting and camera flashes
  - *Preview*: `sports_broadcast.preview.mp4` - Demonstrates sports content processing
  - *Best method*: enhanced_flashcap (92% strength, 0.008 threshold)

#### 🏥 **Medical Warning Content**
- **`medical_warning.mp4`** - ⚠️ **MOST DANGEROUS** - Rapid alternating patterns (medical warning level)
  - *Preview*: `medical_warning.preview.mp4` - Shows heavy pattern suppression
  - *Best method*: enhanced_median (99% strength, 0.001 threshold)

#### 🎭 **Entertainment Content**
- **`concert_lighting.mp4`** - ⚠️ Concert/club with strobe lights and color washes
  - *Preview*: `concert_lighting.preview.mp4` - Shows complex lighting stabilization
  - *Best method*: enhanced_flashcap (95% strength, 0.005 threshold)

- **`animation_effects.mp4`** - ⚠️ Animated content with magic effects and lightning
  - *Preview*: `animation_effects.preview.mp4` - Preserves animation while reducing flashes
  - *Best method*: enhanced_ema (85% strength, 0.01 threshold)

#### 🧪 **Synthetic Test Cases**
- **`flash_test.mp4`** - Simple synthetic test with regular flash pattern
  - *Preview*: `flash_test.preview.mp4` - Basic flicker reduction demonstration
  - *Original synthetic test case*

- **`rapid_strobe.mp4`** - ⚠️ Synthetic rapid strobing test
  - *Preview*: `rapid_strobe.preview.mp4` - Shows rapid strobe suppression
  - *Test case for algorithm validation*

- **`scene_cuts.mp4`** - Video with abrupt scene transitions
  - *Preview*: `scene_cuts.preview.mp4` - Demonstrates transition smoothing
  - *Tests edge case handling*

- **`subtle_flicker.mp4`** - Video with subtle, hard-to-detect flicker patterns
  - *Preview*: `subtle_flicker.preview.mp4` - Shows detection of subtle variations
  - *Tests algorithm sensitivity*

## Usage Examples

### 🎯 **Quick Testing**
```bash
# Test on the most challenging content (porygon)
destrobe run examples/porygon.mp4 --preset maximum

# Preview any test video safely
destrobe preview examples/test_videos/gaming_flicker.mp4

# Analyze flicker levels
destrobe metrics examples/test_videos/medical_warning.mp4

# Batch process all test videos
destrobe run examples/test_videos/ --preset enhanced_strong
```

### 🧪 **Algorithm Testing**
```bash
# Test specific methods on different content types
destrobe run examples/test_videos/gaming_flicker.mp4 --method enhanced_flashcap --strength 0.98
destrobe run examples/test_videos/concert_lighting.mp4 --method enhanced_median --strength 0.95
destrobe run examples/test_videos/animation_effects.mp4 --method enhanced_ema --strength 0.85

# Compare different presets
destrobe run examples/test_videos/medical_warning.mp4 --preset safe
destrobe run examples/test_videos/medical_warning.mp4 --preset maximum
destrobe run examples/test_videos/medical_warning.mp4 --preset nuclear
```

### 📊 **Performance Benchmarking**
```bash
# Benchmark processing speed
destrobe run examples/test_videos/sports_broadcast.mp4 --preset maximum --benchmark

# Compare quality metrics before/after
destrobe metrics examples/test_videos/tv_broadcast.mp4
destrobe metrics examples/test_videos/tv_broadcast.enhanced_flashcap.mp4

# Batch analysis
for video in examples/test_videos/*.mp4; do
  echo "=== $video ==="
  destrobe metrics "$video" --json
done
```

## Results Summary

The `porygon.mp4` example demonstrates destrobe's maximum capabilities:

- **81.1% flicker reduction** achieved with `--preset maximum`
- **89.8% video quality preserved** (SSIM)
- **351.5 fps processing speed**
- Reduces 59 flash events to barely perceptible levels

## Technical Details

- **Resolution**: 320x240 (porygon.mp4)
- **Duration**: 13 seconds
- **Original Flicker Index**: 8.577
- **Processed Flicker Index**: 1.620 (81% reduction)
- **Method**: enhanced_flashcap with 99% strength, 0.001 threshold
