"""
Tests for destrobe filter algorithms.
"""

import numpy as np
import pytest

from destrobe.core.filters import (
    median3_filter,
    ema_filter,
    flashcap_filter,
    apply_filter,
    FrameBuffer,
    bgr_to_yuv,
    yuv_to_bgr,
)


def create_test_frame(height: int = 240, width: int = 320, color: tuple = (0.5, 0.5, 0.5)) -> np.ndarray:
    """Create a test frame with specified color."""
    frame = np.full((height, width, 3), color, dtype=np.float32)
    return frame


def create_flash_sequence() -> list:
    """Create a sequence of frames with a flash in the middle."""
    frames = []
    
    # Dark frame
    frames.append(create_test_frame(color=(0.2, 0.2, 0.2)))
    
    # Bright flash frame
    frames.append(create_test_frame(color=(0.9, 0.9, 0.9)))
    
    # Dark frame again
    frames.append(create_test_frame(color=(0.2, 0.2, 0.2)))
    
    return frames


def create_gradual_sequence() -> list:
    """Create a sequence with gradual brightness changes."""
    frames = []
    
    for brightness in [0.2, 0.4, 0.6]:
        frames.append(create_test_frame(color=(brightness, brightness, brightness)))
    
    return frames


class TestColorConversion:
    """Test color space conversion functions."""
    
    def test_bgr_to_yuv_round_trip(self):
        """Test that BGR -> YUV -> BGR conversion preserves the image."""
        original = create_test_frame()
        
        y, u, v = bgr_to_yuv(original)
        reconstructed = yuv_to_bgr(y, u, v)
        
        # Should be very close (within floating point precision)
        np.testing.assert_allclose(original, reconstructed, rtol=1e-5)
    
    def test_yuv_components_shape(self):
        """Test that YUV components have correct shapes."""
        frame = create_test_frame(height=100, width=200)
        y, u, v = bgr_to_yuv(frame)
        
        assert y.shape == (100, 200)
        assert u.shape == (100, 200)
        assert v.shape == (100, 200)


class TestMedian3Filter:
    """Test median3 filter algorithm."""
    
    def test_median3_flash_suppression(self):
        """Test that median3 filter suppresses flash frames."""
        flash_frames = create_flash_sequence()
        result = median3_filter(flash_frames)
        
        # Result should be closer to the dark frames than the flash
        y_result, _, _ = bgr_to_yuv(result)
        mean_result = np.mean(y_result)
        
        # Should be much darker than the flash frame
        y_flash, _, _ = bgr_to_yuv(flash_frames[1])
        mean_flash = np.mean(y_flash)
        
        assert mean_result < mean_flash * 0.7  # At least 30% reduction
    
    def test_median3_preserves_gradual_changes(self):
        """Test that median3 doesn't over-smooth gradual changes."""
        gradual_frames = create_gradual_sequence()
        result = median3_filter(gradual_frames)
        
        # Middle frame should be approximately preserved
        y_original, _, _ = bgr_to_yuv(gradual_frames[1])
        y_result, _, _ = bgr_to_yuv(result)
        
        mean_diff = np.mean(np.abs(y_original - y_result))
        assert mean_diff < 0.1  # Should be similar
    
    def test_median3_requires_three_frames(self):
        """Test that median3 filter requires exactly 3 frames."""
        with pytest.raises(ValueError):
            median3_filter([create_test_frame()])
        
        with pytest.raises(ValueError):
            median3_filter([create_test_frame(), create_test_frame()])


class TestEMAFilter:
    """Test EMA filter algorithm."""
    
    def test_ema_first_frame(self):
        """Test EMA filter with no previous frame."""
        frame = create_test_frame()
        result = ema_filter(frame, None)
        
        # First frame should be unchanged
        np.testing.assert_array_equal(frame, result)
    
    def test_ema_smoothing(self):
        """Test that EMA filter provides smoothing."""
        dark_frame = create_test_frame(color=(0.2, 0.2, 0.2))
        bright_frame = create_test_frame(color=(0.8, 0.8, 0.8))
        
        result = ema_filter(bright_frame, dark_frame, strength=0.8)
        
        # Result should be between dark and bright
        y_result, _, _ = bgr_to_yuv(result)
        mean_result = np.mean(y_result)
        
        assert 0.2 < mean_result < 0.8
    
    def test_ema_strength_effect(self):
        """Test that higher strength increases smoothing."""
        dark_frame = create_test_frame(color=(0.2, 0.2, 0.2))
        bright_frame = create_test_frame(color=(0.8, 0.8, 0.8))
        
        result_weak = ema_filter(bright_frame, dark_frame, strength=0.1)
        result_strong = ema_filter(bright_frame, dark_frame, strength=0.9)
        
        y_weak, _, _ = bgr_to_yuv(result_weak)
        y_strong, _, _ = bgr_to_yuv(result_strong)
        
        mean_weak = np.mean(y_weak)
        mean_strong = np.mean(y_strong)
        
        # Stronger filtering should be closer to previous frame (darker)
        assert mean_strong < mean_weak


class TestFlashcapFilter:
    """Test flashcap filter algorithm."""
    
    def test_flashcap_detects_flash(self):
        """Test that flashcap filter detects and suppresses flashes."""
        flash_frames = create_flash_sequence()
        result = flashcap_filter(flash_frames, flash_thresh=0.3, strength=0.8)
        
        # Result should be much darker than the flash frame
        y_result, _, _ = bgr_to_yuv(result)
        y_flash, _, _ = bgr_to_yuv(flash_frames[1])
        
        mean_result = np.mean(y_result)
        mean_flash = np.mean(y_flash)
        
        assert mean_result < mean_flash * 0.6  # Significant reduction
    
    def test_flashcap_preserves_gradual(self):
        """Test that flashcap preserves gradual changes."""
        gradual_frames = create_gradual_sequence()
        result = flashcap_filter(gradual_frames, flash_thresh=0.3, strength=0.5)
        
        # Should apply mild temporal blend
        y_original, _, _ = bgr_to_yuv(gradual_frames[1])
        y_result, _, _ = bgr_to_yuv(result)
        
        mean_diff = np.mean(np.abs(y_original - y_result))
        assert mean_diff < 0.15  # Small change for gradual sequences
    
    def test_flashcap_threshold_effect(self):
        """Test that flash threshold affects detection."""
        flash_frames = create_flash_sequence()
        
        # High threshold - should not detect flash
        result_high = flashcap_filter(flash_frames, flash_thresh=0.9, strength=0.8)
        
        # Low threshold - should detect flash
        result_low = flashcap_filter(flash_frames, flash_thresh=0.1, strength=0.8)
        
        y_high, _, _ = bgr_to_yuv(result_high)
        y_low, _, _ = bgr_to_yuv(result_low)
        
        mean_high = np.mean(y_high)
        mean_low = np.mean(y_low)
        
        # Low threshold should suppress more
        assert mean_low < mean_high


class TestApplyFilter:
    """Test the apply_filter wrapper function."""
    
    def test_apply_filter_median3(self):
        """Test apply_filter with median3 method."""
        frames = create_flash_sequence()
        result = apply_filter(frames, "median3")
        
        assert result.shape == frames[0].shape
    
    def test_apply_filter_ema(self):
        """Test apply_filter with ema method."""
        frames = create_gradual_sequence()
        result = apply_filter(frames, "ema", strength=0.5)
        
        assert result.shape == frames[0].shape
    
    def test_apply_filter_flashcap(self):
        """Test apply_filter with flashcap method."""
        frames = create_flash_sequence()
        result = apply_filter(frames, "flashcap", flash_thresh=0.2)
        
        assert result.shape == frames[0].shape
    
    def test_apply_filter_unknown_method(self):
        """Test apply_filter with unknown method."""
        frames = [create_test_frame()]
        
        with pytest.raises(ValueError):
            apply_filter(frames, "unknown_method")
    
    def test_apply_filter_insufficient_frames(self):
        """Test apply_filter with insufficient frames for method."""
        frame = create_test_frame()
        
        # Should return the frame as-is for insufficient frames
        result = apply_filter([frame], "median3")
        np.testing.assert_array_equal(frame, result)


class TestFrameBuffer:
    """Test the FrameBuffer class."""
    
    def test_frame_buffer_median3(self):
        """Test FrameBuffer with median3 method."""
        buffer = FrameBuffer("median3")
        frames = create_flash_sequence()
        
        results = []
        for frame in frames:
            result = buffer.add_frame(frame)
            if result is not None:
                results.append(result)
        
        # Should get one result after adding 3 frames
        assert len(results) == 1
    
    def test_frame_buffer_ema(self):
        """Test FrameBuffer with ema method."""
        buffer = FrameBuffer("ema")
        frames = create_gradual_sequence()
        
        results = []
        for frame in frames:
            result = buffer.add_frame(frame)
            if result is not None:
                results.append(result)
        
        # Should get a result for each frame
        assert len(results) == len(frames)
    
    def test_frame_buffer_flush(self):
        """Test FrameBuffer flush functionality."""
        buffer = FrameBuffer("median3")
        frames = create_flash_sequence()
        
        # Add frames
        for frame in frames:
            buffer.add_frame(frame)
        
        # Flush remaining
        remaining = buffer.flush()
        
        # Should have some remaining frames
        assert len(remaining) > 0
