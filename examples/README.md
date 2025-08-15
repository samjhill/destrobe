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

Located in `test_videos/`:

- **`flash_test.mp4`** - Simple synthetic test with flash pattern
- **`flash_test.preview.mp4`** - Processed preview of flash test
- **`rapid_strobe.mp4`** - Synthetic rapid strobing test
- **`scene_cuts.mp4`** - Video with scene transitions
- **`subtle_flicker.mp4`** - Video with subtle flicker patterns

## Usage Examples

```bash
# Process the main example (safe - creates new file)
destrobe run examples/porygon.mp4 --preset maximum

# Create preview comparison
destrobe preview examples/porygon.mp4 --start 00:00:00

# Analyze flicker metrics
destrobe metrics examples/porygon.mp4

# Batch process all test videos
destrobe run examples/test_videos/ --preset strong
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
