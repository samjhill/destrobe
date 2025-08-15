"""
FPS benchmarking utilities for destrobe.
"""

import platform
import time
from typing import Dict, Any, Optional


def get_system_info() -> Dict[str, str]:
    """Get basic system information for benchmarking."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.architecture()[0],
        'python_version': platform.python_version(),
    }
    
    # Try to get more detailed CPU info on macOS
    if platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info['cpu_model'] = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return info


class FPSCounter:
    """Simple FPS counter for performance measurement."""
    
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.frame_count: int = 0
        self.last_update: Optional[float] = None
        self.current_fps: float = 0.0
    
    def start(self) -> None:
        """Start the FPS counter."""
        self.start_time = time.time()
        self.last_update = self.start_time
        self.frame_count = 0
        self.current_fps = 0.0
    
    def update(self, frames_processed: int = 1) -> None:
        """Update the counter with processed frames."""
        if self.start_time is None:
            self.start()
        
        self.frame_count += frames_processed
        current_time = time.time()
        
        # Update FPS every second
        if current_time - self.last_update >= 1.0:
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.current_fps = self.frame_count / elapsed
            self.last_update = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get complete performance statistics."""
        if self.start_time is None:
            return {'fps': 0.0, 'frames': 0, 'elapsed': 0.0}
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        
        return {
            'fps': fps,
            'frames': self.frame_count,
            'elapsed': elapsed,
        }


def benchmark_system() -> Dict[str, Any]:
    """Run a simple system benchmark."""
    import numpy as np
    
    # CPU benchmark: matrix multiplication
    start_time = time.time()
    
    # Create test matrices
    size = 1000
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)
    
    # Perform multiplication
    _ = np.dot(a, b)
    
    cpu_time = time.time() - start_time
    
    # Memory benchmark: array operations
    start_time = time.time()
    
    arr = np.random.random((10000, 1000)).astype(np.float32)
    result = np.mean(arr, axis=1)
    result = np.std(arr, axis=1)
    
    memory_time = time.time() - start_time
    
    return {
        'cpu_benchmark_ms': cpu_time * 1000,
        'memory_benchmark_ms': memory_time * 1000,
        'system_info': get_system_info(),
    }
