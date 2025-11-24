"""
Persistent monitoring system for long-running Jupyter experiments.

This module provides:
1. File-based logging that persists across client disconnects
2. Progress tracking that writes to JSON files
3. A simple web dashboard to monitor progress remotely
4. TQDM integration for persistent progress bars
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from threading import Lock
import numpy as np

# Try to import tqdm, but don't fail if not available
try:
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None
    tqdm_notebook = None


class PersistentLogger:
    """
    Logger that writes to both console and file, ensuring output persists
    even when Jupyter client disconnects.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Set up logging
        self.logger = logging.getLogger(f"persistent_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler (always write to file)
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (only if connected)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self._lock = Lock()
    
    def info(self, message: str):
        """Log info message."""
        with self._lock:
            self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        with self._lock:
            self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        with self._lock:
            self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        with self._lock:
            self.logger.debug(message)
    
    def print(self, *args, **kwargs):
        """Print that also logs to file."""
        message = ' '.join(str(arg) for arg in args)
        self.info(message)
        # Also print to console if possible
        try:
            print(*args, **kwargs)
        except:
            pass  # If stdout is not available, just log to file


class ProgressTracker:
    """
    Tracks experiment progress and writes to JSON file for remote monitoring.
    """
    
    def __init__(self, status_dir: str, experiment_name: str = "experiment"):
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.status_file = self.status_dir / f"{experiment_name}_status.json"
        self._lock = Lock()
        self._status = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'current_step': None,
            'total_steps': None,
            'progress_percent': 0.0,
            'status': 'running',  # running, completed, error, paused
            'steps': [],
            'messages': [],
            'errors': [],
            'estimated_time_remaining': None,
        }
        self._save_status()
    
    def _save_status(self):
        """Save current status to JSON file."""
        with self._lock:
            self._status['last_update'] = datetime.now().isoformat()
            try:
                with open(self.status_file, 'w') as f:
                    json.dump(self._status, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save status file: {e}")
    
    def set_total_steps(self, total: int):
        """Set total number of steps."""
        with self._lock:
            self._status['total_steps'] = total
            self._save_status()
    
    def update_step(self, step_name: str, step_number: Optional[int] = None, 
                    total_steps: Optional[int] = None, message: Optional[str] = None):
        """Update current step progress."""
        with self._lock:
            if step_number is not None:
                self._status['current_step'] = step_number
            if total_steps is not None:
                self._status['total_steps'] = total_steps
            
            step_info = {
                'name': step_name,
                'step_number': step_number or self._status.get('current_step'),
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            self._status['steps'].append(step_info)
            
            # Keep only last 100 steps to avoid file bloat
            if len(self._status['steps']) > 100:
                self._status['steps'] = self._status['steps'][-100:]
            
            # Calculate progress
            if self._status['total_steps'] and step_number is not None:
                self._status['progress_percent'] = (step_number / self._status['total_steps']) * 100
            
            if message:
                self._status['messages'].append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message
                })
                # Keep only last 50 messages
                if len(self._status['messages']) > 50:
                    self._status['messages'] = self._status['messages'][-50:]
            
            self._save_status()
    
    def add_message(self, message: str, level: str = 'info'):
        """Add a status message."""
        with self._lock:
            self._status['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            })
            if len(self._status['messages']) > 50:
                self._status['messages'] = self._status['messages'][-50:]
            self._save_status()
    
    def add_error(self, error_message: str):
        """Add an error message."""
        with self._lock:
            self._status['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'message': error_message
            })
            self._status['status'] = 'error'
            if len(self._status['errors']) > 20:
                self._status['errors'] = self._status['errors'][-20:]
            self._save_status()
    
    def complete(self, message: Optional[str] = None):
        """Mark experiment as completed."""
        with self._lock:
            self._status['status'] = 'completed'
            self._status['progress_percent'] = 100.0
            if message:
                self.add_message(message)
            self._save_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        with self._lock:
            return self._status.copy()


class PersistentTQDM:
    """
    Wrapper for tqdm that also writes progress to file.
    """
    
    def __init__(self, iterable=None, desc=None, total=None, 
                 progress_tracker: Optional[ProgressTracker] = None,
                 logger: Optional[PersistentLogger] = None,
                 **kwargs):
        self.progress_tracker = progress_tracker
        self.logger = logger
        self.desc = desc
        self.total = total or (len(iterable) if iterable else None)
        self.current = 0
        
        # Create tqdm instance
        if TQDM_AVAILABLE:
            # Try notebook version first, fall back to regular
            try:
                self.tqdm = tqdm_notebook(iterable, desc=desc, total=total, **kwargs)
            except:
                self.tqdm = tqdm(iterable, desc=desc, total=total, **kwargs)
        else:
            self.tqdm = None
            self.iterable = iterable
            self._iter = iter(iterable) if iterable else None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.tqdm:
            try:
                value = next(self.tqdm)
                self._update_progress()
                return value
            except StopIteration:
                self._finalize()
                raise
        else:
            if self._iter is None:
                raise StopIteration
            try:
                value = next(self._iter)
                self.current += 1
                self._update_progress()
                return value
            except StopIteration:
                self._finalize()
                raise
    
    def update(self, n=1):
        """Update progress by n steps."""
        if self.tqdm:
            self.tqdm.update(n)
        self.current += n
        self._update_progress()
    
    def _update_progress(self):
        """Update progress tracker and logger."""
        if self.progress_tracker:
            progress_pct = (self.current / self.total * 100) if self.total else 0
            self.progress_tracker.update_step(
                step_name=self.desc or "Progress",
                step_number=self.current,
                total_steps=self.total,
                message=f"{self.desc}: {self.current}/{self.total} ({progress_pct:.1f}%)"
            )
        
        if self.logger and self.current % max(1, self.total // 20) == 0:  # Log every 5%
            progress_pct = (self.current / self.total * 100) if self.total else 0
            self.logger.info(f"{self.desc}: {self.current}/{self.total} ({progress_pct:.1f}%)")
    
    def _finalize(self):
        """Finalize progress tracking."""
        if self.progress_tracker:
            self.progress_tracker.update_step(
                step_name=self.desc or "Progress",
                step_number=self.current,
                total_steps=self.total,
                message=f"{self.desc}: Completed"
            )
        if self.logger:
            self.logger.info(f"{self.desc}: Completed ({self.current}/{self.total})")
    
    def close(self):
        """Close the progress bar."""
        if self.tqdm:
            self.tqdm.close()
        self._finalize()


class ExperimentMonitor:
    """
    Main class for monitoring long-running experiments.
    Combines logging, progress tracking, and status reporting.
    """
    
    def __init__(self, base_dir: str, experiment_name: str = "experiment"):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        
        # Create subdirectories
        self.log_dir = self.base_dir / "logs"
        self.status_dir = self.base_dir / "status"
        self.plots_dir = self.base_dir / "plots"
        
        for d in [self.log_dir, self.status_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = PersistentLogger(self.log_dir, experiment_name)
        self.progress = ProgressTracker(self.status_dir, experiment_name)
        
        self.logger.info(f"Experiment monitor initialized: {experiment_name}")
        self.logger.info(f"Log file: {self.logger.log_file}")
        self.logger.info(f"Status file: {self.progress.status_file}")
    
    def tqdm(self, iterable=None, desc=None, total=None, **kwargs):
        """Create a persistent tqdm progress bar."""
        return PersistentTQDM(
            iterable=iterable,
            desc=desc,
            total=total,
            progress_tracker=self.progress,
            logger=self.logger,
            **kwargs
        )
    
    def log(self, message: str, level: str = 'info'):
        """Log a message."""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
            self.progress.add_error(message)
        elif level == 'debug':
            self.logger.debug(message)
        
        self.progress.add_message(message, level)
    
    def update_progress(self, step_name: str, step_number: Optional[int] = None,
                       total_steps: Optional[int] = None, message: Optional[str] = None):
        """Update experiment progress."""
        self.progress.update_step(step_name, step_number, total_steps, message)
        if message:
            self.logger.info(f"{step_name}: {message}")
    
    def complete(self, message: Optional[str] = None):
        """Mark experiment as complete."""
        self.progress.complete(message)
        if message:
            self.logger.info(message)
        self.logger.info("Experiment completed successfully")
    
    def save_plot(self, fig, filename: str):
        """Save a matplotlib figure to the plots directory."""
        plot_path = self.plots_dir / filename
        try:
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Plot saved: {plot_path}")
            # Update status with plot info
            self.progress.add_message(f"Plot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save plot {filename}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        status = self.progress.get_status()
        status['log_file'] = str(self.logger.log_file)
        status['plots_dir'] = str(self.plots_dir)
        return status


def create_monitor(base_dir: str, experiment_name: str = "experiment") -> ExperimentMonitor:
    """
    Convenience function to create an experiment monitor.
    
    Usage:
        monitor = create_monitor(expt_path, "autocalibration_20250101")
        for i in monitor.tqdm(range(100), desc="Running experiments"):
            # do work
            monitor.log(f"Completed step {i}")
    """
    return ExperimentMonitor(base_dir, experiment_name)









