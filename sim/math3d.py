"""Quaternion helpers for 3D rigid body simulation."""

from __future__ import annotations

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return normalized quaternion. Falls back to identity on zero norm."""
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q (body -> world)."""
    qn = quat_normalize(q)
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    q_conj = np.array([qn[0], -qn[1], -qn[2], -qn[3]], dtype=float)
    rotated = quat_mul(quat_mul(qn, v_quat), q_conj)
    return rotated[1:]
