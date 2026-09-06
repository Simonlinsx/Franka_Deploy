"""Perception-only AnyDexGrasp bridge for calibrated RGB-D point clouds.

The package deliberately contains no robot or dexterous-hand transport code.
Importing it never opens a camera, serial port, or robot connection.
"""

from .types import GraspCandidate, GraspResult, PointCloudObservation

__all__ = ["GraspCandidate", "GraspResult", "PointCloudObservation"]

__version__ = "0.1.0"
