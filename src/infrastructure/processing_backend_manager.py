"""Lightweight processing backend manager for image processing workflow."""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("MammographyCAD.backend")


class ProcessingStatus(Enum):
    """Enumeration of processing states."""
    IDLE = "Idle"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    status: ProcessingStatus = ProcessingStatus.IDLE
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    last_operation: str = ""
    error_message: Optional[str] = None


class ProcessingBackendManager:
    """
    Lightweight processing backend manager.
    
    Manages the image processing workflow and provides a centralized interface
    for processing modules. Designed for future extensibility without GPU-specific
    features.
    
    Responsibilities:
        - Manage image processing workflow
        - Provide centralized interface for processing modules
        - Store and retrieve processing mode
        - Monitor processing status (Idle, Running, Completed, Failed)
        - Provide processing statistics (execution time, status, last operation)
    
    Design Principles:
        - CPU-only implementation
        - No GPU dependencies (CUDA, OpenCL, PyTorch, etc.)
        - Extensible for future GPU acceleration
        - Clean, modular, well-documented code
    
    Example:
        >>> manager = ProcessingBackendManager.instance()
        >>> manager.initialize()
        >>> manager.set_processing_mode("high_quality")
        >>> manager.start_processing("mammogram_analysis")
        >>> # ... processing occurs ...
        >>> manager.finish_processing(success=True)
        >>> stats = manager.get_processing_status()
    """
    
    _instance: Optional["ProcessingBackendManager"] = None
    
    # Default processing modes (can be extended in future)
    VALID_MODES = ["standard", "fast", "high_quality", "batch"]

    def __init__(self):
        """Initialize the ProcessingBackendManager."""
        self._processing_mode: str = "standard"
        self._status: ProcessingStatus = ProcessingStatus.IDLE
        self._stats: ProcessingStats = ProcessingStats()
        self._initialized: bool = False
        self._start_timestamp: Optional[float] = None

    @classmethod
    def instance(cls) -> "ProcessingBackendManager":
        """
        Get singleton instance of ProcessingBackendManager.
        
        Returns:
            ProcessingBackendManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        """
        Initialize the processing backend manager.
        
        Sets up the manager for use. Should be called before other operations.
        Idempotent - can be called multiple times safely.
        """
        if self._initialized:
            logger.debug("ProcessingBackendManager already initialized")
            return
        
        self._initialized = True
        self._status = ProcessingStatus.IDLE
        self._stats = ProcessingStats()
        self._processing_mode = "standard"
        logger.info("ProcessingBackendManager initialized successfully")

    def start_processing(self, operation: str = "") -> bool:
        """
        Start a processing operation.
        
        Args:
            operation: Description of the operation being started (optional).
        
        Returns:
            bool: True if processing started successfully, False otherwise.
        
        Raises:
            RuntimeError: If manager is not initialized.
        """
        if not self._initialized:
            logger.error("Cannot start processing: manager not initialized")
            raise RuntimeError("ProcessingBackendManager must be initialized before use")
        
        if self._status == ProcessingStatus.RUNNING:
            logger.warning("Processing already in progress (status: %s)", self._status.value)
            return False
        
        self._status = ProcessingStatus.RUNNING
        self._stats.status = ProcessingStatus.RUNNING
        self._stats.start_time = time.time()
        self._stats.last_operation = operation
        self._stats.error_message = None
        self._start_timestamp = time.time()
        
        logger.info(
            "Processing started: operation='%s', mode='%s'",
            operation or "unnamed",
            self._processing_mode
        )
        return True

    def finish_processing(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """
        Finish a processing operation.
        
        Args:
            success: Whether the processing completed successfully.
            error_message: Optional error message if processing failed.
        
        Raises:
            RuntimeError: If manager is not initialized or no processing is running.
        """
        if not self._initialized:
            logger.error("Cannot finish processing: manager not initialized")
            raise RuntimeError("ProcessingBackendManager must be initialized before use")
        
        if self._status != ProcessingStatus.RUNNING:
            logger.warning(
                "Cannot finish processing: current status is %s (not running)",
                self._status.value
            )
            return
        
        end_time = time.time()
        self._stats.end_time = end_time
        self._stats.execution_time = end_time - self._stats.start_time
        
        if success:
            self._status = ProcessingStatus.COMPLETED
            self._stats.status = ProcessingStatus.COMPLETED
            logger.info(
                "Processing completed: operation='%s', time=%.3fs, mode='%s'",
                self._stats.last_operation,
                self._stats.execution_time,
                self._processing_mode
            )
        else:
            self._status = ProcessingStatus.FAILED
            self._stats.status = ProcessingStatus.FAILED
            self._stats.error_message = error_message or "Unknown error occurred"
            logger.error(
                "Processing failed: operation='%s', error='%s', mode='%s'",
                self._stats.last_operation,
                self._stats.error_message,
                self._processing_mode
            )

    def set_processing_mode(self, mode: str) -> bool:
        """
        Set the processing mode.
        
        Args:
            mode: The processing mode to set (e.g., 'standard', 'fast', 'high_quality').
        
        Returns:
            bool: True if mode was set successfully, False if mode is invalid.
        
        Note:
            Mode validation is minimal to allow future extensibility.
            Invalid modes will be logged but not rejected strictly.
        """
        if not self._initialized:
            logger.error("Cannot set processing mode: manager not initialized")
            return False
        
        if mode not in self.VALID_MODES:
            logger.warning(
                "Unknown processing mode '%s'. Valid modes: %s",
                mode,
                ", ".join(self.VALID_MODES)
            )
            # Don't reject unknown modes to allow future extensibility
        
        old_mode = self._processing_mode
        self._processing_mode = mode
        logger.info("Processing mode changed: '%s' -> '%s'", old_mode, mode)
        return True

    def get_processing_mode(self) -> str:
        """
        Get the current processing mode.
        
        Returns:
            str: The current processing mode.
        """
        return self._processing_mode

    def get_processing_status(self) -> ProcessingStatus:
        """
        Get the current processing status.
        
        Returns:
            ProcessingStatus: Current status (Idle, Running, Completed, Failed).
        """
        return self._status

    def get_last_processing_time(self) -> float:
        """
        Get the execution time of the last processing operation.
        
        Returns:
            float: Execution time in seconds. Returns 0.0 if no processing has occurred.
        """
        return self._stats.execution_time

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.
        
        Returns:
            Dict containing:
                - status: Current processing status
                - mode: Current processing mode
                - execution_time: Last operation execution time (seconds)
                - last_operation: Description of last operation
                - start_time: Timestamp when last operation started
                - end_time: Timestamp when last operation ended
                - error_message: Error message if last operation failed
        """
        return {
            "status": self._status.value,
            "mode": self._processing_mode,
            "execution_time": self._stats.execution_time,
            "last_operation": self._stats.last_operation,
            "start_time": self._stats.start_time,
            "end_time": self._stats.end_time,
            "error_message": self._stats.error_message,
        }

    def reset(self) -> None:
        """
        Reset the processing backend manager to initial state.
        
        All processing state is cleared. The manager must be re-initialized
        before use after reset.
        """
        self._processing_mode = "standard"
        self._status = ProcessingStatus.IDLE
        self._stats = ProcessingStats()
        self._initialized = False
        self._start_timestamp = None
        
        logger.info("ProcessingBackendManager reset to initial state")

    def is_initialized(self) -> bool:
        """
        Check if the manager has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._initialized

    def __repr__(self) -> str:
        """String representation of the manager."""
        return (
            f"ProcessingBackendManager(mode='{self._processing_mode}', "
            f"status={self._status.value}, initialized={self._initialized})"
        )