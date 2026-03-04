"""
EsdfMapAdapter: Wraps esdf/query ROS2 service for APF obstacle avoidance.
Provides get_esdf, compute_gradient, is_collision_line_segment interface
used by control_law_3D and Planning_3D.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import rclpy
from rclpy.node import Node

try:
    from esdf_map.srv import QueryEsdf
except ImportError:
    QueryEsdf = None


class EsdfMapAdapter:
    """
    Adapter for esdf/query service. Exposes origin, dims, resolution and
    get_esdf/compute_gradient/is_collision_line_segment for APF planners.
    """

    def __init__(
        self,
        node: Node,
        service_name: str = "esdf/query",
        frame_id: str = "map_origin",
        map_origin_x: float = -5.0,
        map_origin_y: float = -7.5,
        map_origin_z: float = 0.0,
        map_size_x: float = 22.0,
        map_size_y: float = 17.0,
        map_size_z: float = 6.0,
        resolution: float = 0.15,
    ) -> None:
        self._node = node
        self._service_name = service_name
        self._frame_id = frame_id
        self.origin = (map_origin_x, map_origin_y, map_origin_z)
        nx = max(1, int(round(map_size_x / resolution)))
        ny = max(1, int(round(map_size_y / resolution)))
        nz = max(1, int(round(map_size_z / resolution)))
        self.dims = np.array([nx, ny, nz], dtype=int)
        self.resolution = resolution

        if QueryEsdf is None:
            node.get_logger().warn(
                "esdf_map.srv.QueryEsdf not found; ESDF queries will return defaults"
            )
            self._client = None
            return

        self._client = node.create_client(QueryEsdf, service_name)
        if not self._client.wait_for_service(timeout_sec=5.0):
            node.get_logger().warn(
                f"esdf/query service '{service_name}' not available; "
                "obstacle avoidance may be disabled"
            )

    def _query(self, x: float, y: float, z: float) -> Optional[tuple[float, np.ndarray]]:
        """Query esdf/query service. Returns (distance, gradient) or None."""
        if self._client is None or QueryEsdf is None:
            return (5.0, np.zeros(3))  # safe default

        if not self._client.service_is_ready():
            return (5.0, np.zeros(3))

        req = QueryEsdf.Request()
        req.position.x = float(x)
        req.position.y = float(y)
        req.position.z = float(z)
        req.frame_id = self._frame_id

        try:
            future = self._client.call_async(req)
            rclpy.spin_until_future_complete(
                self._node, future, timeout_sec=1.0
            )
            if not future.done():
                return None
            resp = future.result()
            if resp is None or not resp.success:
                return None
            grad = np.array([resp.gradient.x, resp.gradient.y, resp.gradient.z])
            return (float(resp.distance), grad)
        except Exception:
            return None

    def get_esdf(self, pos: Union[np.ndarray, list, tuple]) -> Union[float, np.ndarray]:
        """
        Get ESDF distance at position(s).
        pos: (3,) or (N, 3) array. Returns float or (N,) array.
        """
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        n = pos.shape[0]
        dists = np.full(n, 5.0)  # default safe distance
        for i in range(n):
            r = self._query(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            if r is not None:
                dists[i] = r[0]
        return float(dists[0]) if n == 1 else dists

    def compute_gradient(self, pos: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        """Get ESDF gradient at position. Returns (3,) array or None."""
        pos = np.asarray(pos, dtype=float).flatten()
        if len(pos) < 3:
            return None
        r = self._query(float(pos[0]), float(pos[1]), float(pos[2]))
        if r is None:
            return None
        return r[1]

    def is_collision_line_segment(
        self,
        point1: Union[np.ndarray, list, tuple],
        point2: Union[np.ndarray, list, tuple],
        safe_margin: float = 0.2,
        num_samples: int = 10,
    ) -> bool:
        """
        Check if line segment from point1 to point2 collides with obstacles.
        Samples along the segment and returns True if any sample has distance < safe_margin.
        """
        p1 = np.asarray(point1, dtype=float).flatten()[:3]
        p2 = np.asarray(point2, dtype=float).flatten()[:3]
        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            r = self._query(float(pt[0]), float(pt[1]), float(pt[2]))
            if r is None:
                continue
            if r[0] < safe_margin:
                return True
        return False
