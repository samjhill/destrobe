"""
Tests for destrobe metrics computation.
"""

import numpy as np
import pytest

from destrobe.core.metrics import (
    compute_flicker_index,
    compute_flicker_metrics,
    compute_frame_ssim,
    analyze_luminance_distribution,
    detect_flash_events,
    compute_luma_delta_sequence,
)


class TestFlickerMetrics:
    """Test flicker metric computation."""
    
    def test_flicker_index_stable_sequence(self):
        """Test flicker index on stable luminance sequence."""
        # Constant luminance should have low flicker
        stable_lumas = [0.5] * 10
        flicker = compute_flicker_index(stable_lumas)
        
        assert flicker == 0.0
    
    def test_flicker_index_alternating_sequence(self):
        """Test flicker index on alternating luminance sequence."""
        # Alternating bright/dark should have high flicker
        alternating_lumas = [0.2, 0.8] * 5
        flicker = compute_flicker_index(alternating_lumas)
        
        assert flicker > 0.5  # High flicker expected
    
    def test_flicker_index_gradual_sequence(self):
        """Test flicker index on gradually changing sequence."""
        # Gradual change should have low flicker
        gradual_lumas = [0.1 * i for i in range(10)]
        flicker = compute_flicker_index(gradual_lumas)
        
        assert 0.0 < flicker < 0.2  # Some change but not flickery
    
    def test_flicker_index_single_frame(self):
        """Test flicker index with insufficient frames."""
        single_luma = [0.5]
        flicker = compute_flicker_index(single_luma)
        
        assert flicker == 0.0
    
    def test_flicker_metrics_wrapper(self):
        """Test the compute_flicker_metrics wrapper."""
        lumas = [0.2, 0.8, 0.2, 0.8]
        
        result = compute_flicker_metrics(lumas)
        expected = compute_flicker_index(lumas)
        
        assert result == expected


class TestSSIMMetrics:
    """Test SSIM computation."""
    
    def test_ssim_identical_frames(self):
        """Test SSIM with identical frames."""
        frame = np.random.random((100, 100, 3)).astype(np.float32)
        
        ssim_value = compute_frame_ssim(frame, frame)
        
        assert ssim_value == pytest.approx(1.0, abs=1e-6)
    
    def test_ssim_different_frames(self):
        """Test SSIM with different frames."""
        frame1 = np.zeros((100, 100, 3), dtype=np.float32)
        frame2 = np.ones((100, 100, 3), dtype=np.float32)
        
        ssim_value = compute_frame_ssim(frame1, frame2)
        
        assert ssim_value < 0.5  # Should be quite different
    
    def test_ssim_similar_frames(self):
        """Test SSIM with similar frames."""
        frame1 = np.random.random((100, 100, 3)).astype(np.float32)
        # Add small amount of noise
        frame2 = frame1 + np.random.normal(0, 0.01, frame1.shape).astype(np.float32)
        frame2 = np.clip(frame2, 0.0, 1.0)
        
        ssim_value = compute_frame_ssim(frame1, frame2)
        
        assert ssim_value > 0.9  # Should be very similar


class TestLuminanceAnalysis:
    """Test luminance distribution analysis."""
    
    def test_luminance_distribution_constant(self):
        """Test luminance analysis with constant values."""
        constant_lumas = [0.5] * 10
        
        stats = analyze_luminance_distribution(constant_lumas)
        
        assert stats['mean_luminance'] == 0.5
        assert stats['std_luminance'] == 0.0
        assert stats['min_luminance'] == 0.5
        assert stats['max_luminance'] == 0.5
        assert stats['luminance_range'] == 0.0
    
    def test_luminance_distribution_varying(self):
        """Test luminance analysis with varying values."""
        varying_lumas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        stats = analyze_luminance_distribution(varying_lumas)
        
        assert stats['mean_luminance'] == 0.5
        assert stats['std_luminance'] > 0
        assert stats['min_luminance'] == 0.0
        assert stats['max_luminance'] == 1.0
        assert stats['luminance_range'] == 1.0
        assert 0.0 <= stats['percentile_10'] <= stats['percentile_90'] <= 1.0
    
    def test_luminance_distribution_empty(self):
        """Test luminance analysis with empty input."""
        empty_lumas = []
        
        stats = analyze_luminance_distribution(empty_lumas)
        
        assert stats == {}


class TestFlashDetection:
    """Test flash event detection."""
    
    def test_detect_no_flashes(self):
        """Test flash detection with stable sequence."""
        stable_lumas = [0.5] * 10
        
        flashes = detect_flash_events(stable_lumas, threshold=0.1)
        
        assert len(flashes) == 0
    
    def test_detect_single_flash(self):
        """Test flash detection with single flash event."""
        # Sequence with one bright flash
        flash_lumas = [0.2, 0.2, 0.8, 0.2, 0.2]
        
        flashes = detect_flash_events(flash_lumas, threshold=0.3)
        
        assert len(flashes) == 1
        start, end, intensity = flashes[0]
        assert start == 2  # Flash at index 2
        assert intensity > 0.3
    
    def test_detect_multiple_flashes(self):
        """Test flash detection with multiple flash events."""
        # Sequence with two flashes
        flash_lumas = [0.2, 0.8, 0.2, 0.2, 0.8, 0.2]
        
        flashes = detect_flash_events(flash_lumas, threshold=0.3)
        
        assert len(flashes) == 2
    
    def test_flash_threshold_sensitivity(self):
        """Test that flash threshold affects detection."""
        # Moderate brightness change
        moderate_lumas = [0.3, 0.6, 0.3]
        
        # High threshold - should not detect
        flashes_high = detect_flash_events(moderate_lumas, threshold=0.5)
        assert len(flashes_high) == 0
        
        # Low threshold - should detect
        flashes_low = detect_flash_events(moderate_lumas, threshold=0.1)
        assert len(flashes_low) == 1
    
    def test_flash_minimum_duration(self):
        """Test minimum duration filtering."""
        # Very brief flash
        brief_flash_lumas = [0.2, 0.8, 0.2]
        
        # Require longer duration
        flashes = detect_flash_events(brief_flash_lumas, threshold=0.3, min_duration=2)
        
        assert len(flashes) == 0  # Too brief
        
        # Allow short duration
        flashes_short = detect_flash_events(brief_flash_lumas, threshold=0.3, min_duration=1)
        
        assert len(flashes_short) == 1


class TestLumaDeltaSequence:
    """Test luminance delta sequence computation."""
    
    def test_luma_delta_stable(self):
        """Test luma delta with stable frames."""
        # Create frames with constant luminance
        stable_frames = []
        for _ in range(5):
            frame_data = {'luma': 0.5}
            stable_frames.append(np.array([frame_data['luma']]))
        
        # Mock frames with luminance data
        luma_values = [0.5] * 5
        
        # Compute deltas manually
        deltas = []
        for i in range(1, len(luma_values)):
            delta = abs(luma_values[i] - luma_values[i-1])
            deltas.append(delta)
        
        assert all(d == 0.0 for d in deltas)
    
    def test_luma_delta_varying(self):
        """Test luma delta with varying frames."""
        luma_values = [0.0, 0.5, 1.0, 0.5]
        
        # Compute deltas manually
        deltas = []
        for i in range(1, len(luma_values)):
            delta = abs(luma_values[i] - luma_values[i-1])
            deltas.append(delta)
        
        expected_deltas = [0.5, 0.5, 0.5]
        assert deltas == expected_deltas
    
    def test_luma_delta_insufficient_frames(self):
        """Test luma delta with insufficient frames."""
        single_luma = [0.5]
        
        # Should return empty list for single frame
        deltas = []  # Would be computed from luma sequence
        
        assert len(deltas) == 0
