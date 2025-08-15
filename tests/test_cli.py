"""
Tests for destrobe CLI interface.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from destrobe.cli import app, discover_video_files, check_first_run, PRESETS


class TestVideoFileDiscovery:
    """Test video file discovery functionality."""
    
    def test_discover_single_file(self, tmp_path):
        """Test discovering a single video file."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        
        files = discover_video_files([video_file])
        
        assert len(files) == 1
        assert files[0] == video_file
    
    def test_discover_non_video_file(self, tmp_path):
        """Test that non-video files are ignored."""
        text_file = tmp_path / "test.txt"
        text_file.touch()
        
        files = discover_video_files([text_file])
        
        assert len(files) == 0
    
    def test_discover_directory_non_recursive(self, tmp_path):
        """Test discovering files in directory (non-recursive)."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        sub_video = subdir / "sub.mp4"
        sub_video.touch()
        
        files = discover_video_files([tmp_path], recursive=False)
        
        assert len(files) == 1
        assert video_file in files
        assert sub_video not in files
    
    def test_discover_directory_recursive(self, tmp_path):
        """Test discovering files in directory (recursive)."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        sub_video = subdir / "sub.mp4"
        sub_video.touch()
        
        files = discover_video_files([tmp_path], recursive=True)
        
        assert len(files) == 2
        assert video_file in files
        assert sub_video in files
    
    def test_discover_mixed_extensions(self, tmp_path):
        """Test discovering files with different video extensions."""
        extensions = [".mp4", ".mkv", ".avi", ".mov"]
        expected_files = []
        
        for ext in extensions:
            video_file = tmp_path / f"test{ext}"
            video_file.touch()
            expected_files.append(video_file)
        
        files = discover_video_files([tmp_path])
        
        assert len(files) == len(extensions)
        for expected in expected_files:
            assert expected in files


class TestFirstRunCheck:
    """Test first run warning functionality."""
    
    def test_first_run_no_warn(self):
        """Test that no_warn flag suppresses warning."""
        # Should not raise any exception or print anything
        check_first_run(no_warn=True)
    
    @patch('destrobe.cli.FIRST_RUN_FILE')
    @patch('destrobe.cli.CONFIG_DIR')
    def test_first_run_creates_marker(self, mock_config_dir, mock_first_run_file):
        """Test that first run creates marker file."""
        mock_config_dir.mkdir = MagicMock()
        mock_first_run_file.exists.return_value = False
        mock_first_run_file.touch = MagicMock()
        
        check_first_run(no_warn=False)
        
        mock_config_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_first_run_file.touch.assert_called_once()
    
    @patch('destrobe.cli.FIRST_RUN_FILE')
    def test_subsequent_run_no_warning(self, mock_first_run_file):
        """Test that subsequent runs don't show warning."""
        mock_first_run_file.exists.return_value = True
        
        # Should not create any new files
        check_first_run(no_warn=False)


class TestCLICommands:
    """Test CLI command functionality."""
    
    def setUp(self):
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "destrobe" in result.stdout
    
    def test_run_command_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "Process video files" in result.stdout
    
    def test_preview_command_help(self):
        """Test preview command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["preview", "--help"])
        
        assert result.exit_code == 0
        assert "side-by-side preview" in result.stdout
    
    def test_metrics_command_help(self):
        """Test metrics command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["metrics", "--help"])
        
        assert result.exit_code == 0
        assert "flicker metrics" in result.stdout
    
    @patch('destrobe.cli.VideoProcessor')
    @patch('destrobe.cli.discover_video_files')
    @patch('destrobe.cli.check_first_run')
    def test_run_command_basic(self, mock_check_first_run, mock_discover, mock_processor):
        """Test basic run command functionality."""
        runner = CliRunner()
        
        # Mock file discovery
        mock_video_file = Path("test.mp4")
        mock_discover.return_value = [mock_video_file]
        
        # Mock processor
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.process_video.return_value = {
            "flicker_before": 0.1,
            "flicker_after": 0.05,
            "ssim": 0.95,
            "fps": 60.0
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "run", "test.mp4",
                "--outdir", tmp_dir,
                "--no-warn"
            ])
        
        # Should not exit with error
        assert result.exit_code == 0
        mock_check_first_run.assert_called_once()
    
    def test_run_command_invalid_method(self):
        """Test run command with invalid method."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "run", "test.mp4",
            "--method", "invalid_method",
            "--no-warn"
        ])
        
        assert result.exit_code == 1
        assert "Unknown method" in result.stdout
    
    def test_run_command_invalid_strength(self):
        """Test run command with invalid strength."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "run", "test.mp4",
            "--strength", "2.0",  # Invalid: > 1.0
            "--no-warn"
        ])
        
        assert result.exit_code == 1
        assert "Strength must be between" in result.stdout
    
    def test_preset_application(self):
        """Test that presets are applied correctly."""
        runner = CliRunner()
        
        # Test with valid preset
        with patch('destrobe.cli.VideoProcessor') as mock_processor, \
             patch('destrobe.cli.discover_video_files') as mock_discover, \
             patch('destrobe.cli.check_first_run'):
            
            mock_discover.return_value = []  # No files to avoid processing
            
            result = runner.invoke(app, [
                "run", "test.mp4",
                "--preset", "safe",
                "--no-warn"
            ])
            
            # Should use preset values
            mock_processor.assert_called_once()
            call_args = mock_processor.call_args[1]
            assert call_args['method'] == PRESETS['safe']['method']
            assert call_args['strength'] == PRESETS['safe']['strength']
    
    def test_invalid_preset(self):
        """Test run command with invalid preset."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "run", "test.mp4",
            "--preset", "invalid_preset",
            "--no-warn"
        ])
        
        assert result.exit_code == 1
        assert "Unknown preset" in result.stdout
    
    @patch('destrobe.cli.create_preview')
    def test_preview_command_basic(self, mock_create_preview):
        """Test basic preview command functionality."""
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            result = runner.invoke(app, [
                "preview", str(tmp_path)
            ])
            
            # Should call create_preview
            mock_create_preview.assert_called_once()
    
    def test_preview_command_missing_file(self):
        """Test preview command with missing file."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "preview", "nonexistent.mp4"
        ])
        
        assert result.exit_code == 1
        assert "File not found" in result.stdout
    
    @patch('destrobe.cli.compute_video_metrics')
    def test_metrics_command_basic(self, mock_compute_metrics):
        """Test basic metrics command functionality."""
        runner = CliRunner()
        
        mock_compute_metrics.return_value = {
            "flicker_index": 0.05,
            "total_frames": 1000,
            "fps": 30.0
        }
        
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            result = runner.invoke(app, [
                "metrics", str(tmp_path)
            ])
            
            assert result.exit_code == 0
            mock_compute_metrics.assert_called_once()
    
    @patch('destrobe.cli.compute_video_metrics')
    def test_metrics_command_json_output(self, mock_compute_metrics):
        """Test metrics command with JSON output."""
        runner = CliRunner()
        
        mock_metrics = {
            "flicker_index": 0.05,
            "total_frames": 1000,
            "fps": 30.0
        }
        mock_compute_metrics.return_value = mock_metrics
        
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            result = runner.invoke(app, [
                "metrics", str(tmp_path), "--json"
            ])
            
            assert result.exit_code == 0
            # Should contain JSON-formatted output
            assert '"flicker_index"' in result.stdout
            assert "0.05" in result.stdout
    
    def test_metrics_command_missing_file(self):
        """Test metrics command with missing file."""
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "metrics", "nonexistent.mp4"
        ])
        
        assert result.exit_code == 1
        assert "File not found" in result.stdout
