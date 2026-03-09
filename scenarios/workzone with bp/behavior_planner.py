"""
Behavior planner for the workzone-with-bp scenario.

Rule implemented:
- When the ego reaches the configured trigger x-position, lock the scenario's
  final route onto the lane that contains the waypoint closest to the configured
  lane-change anchor point.
"""

from __future__ import annotations

import math
from typing import List, Mapping, Sequence, Tuple


class WorkzoneBehaviorPlanner:
    """Stateful planner for the workzone lane-change trigger."""

    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.trigger_x_m = float(cfg.get("trigger_x_m", -20.0))
        self.trigger_when_ego_x_ge = bool(cfg.get("trigger_when_ego_x_ge", True))

        lane_change_anchor_xy = list(cfg.get("lane_change_anchor_xy", [9.73, -2.93]))
        while len(lane_change_anchor_xy) < 2:
            lane_change_anchor_xy.append(0.0)
        self.lane_change_anchor_xy = (
            float(lane_change_anchor_xy[0]),
            float(lane_change_anchor_xy[1]),
        )

        self._locked_destination: List[float] | None = None

    @property
    def is_triggered(self) -> bool:
        return self._locked_destination is not None

    @staticmethod
    def _position_of_waypoint(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position_raw = waypoint.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            return None
        return float(position_raw[0]), float(position_raw[1])

    def _closest_waypoint(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        target_x_m: float,
        target_y_m: float,
    ) -> Mapping[str, object] | None:
        valid_waypoints: List[Mapping[str, object]] = []
        for waypoint in lane_center_waypoints:
            if self._position_of_waypoint(waypoint) is None:
                continue
            valid_waypoints.append(waypoint)
        if len(valid_waypoints) == 0:
            return None

        return min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(target_x_m),
                float(self._position_of_waypoint(waypoint)[1]) - float(target_y_m),
            ),
        )

    def _project_destination_to_lane(
        self,
        current_final_destination_state: Sequence[float],
        lane_center_waypoints: Sequence[Mapping[str, object]],
        target_lane_id: int,
    ) -> List[float] | None:
        valid_lane_waypoints: List[Mapping[str, object]] = []
        for waypoint in lane_center_waypoints:
            if int(waypoint.get("lane_id", -1)) != int(target_lane_id):
                continue
            if self._position_of_waypoint(waypoint) is None:
                continue
            valid_lane_waypoints.append(waypoint)
        if len(valid_lane_waypoints) == 0:
            return None

        target_x_m = float(current_final_destination_state[0]) if len(current_final_destination_state) >= 1 else 0.0
        projected_waypoint = min(
            valid_lane_waypoints,
            key=lambda waypoint: abs(float(self._position_of_waypoint(waypoint)[0]) - float(target_x_m)),
        )
        projected_position = self._position_of_waypoint(projected_waypoint)
        if projected_position is None:
            return None

        projected_heading_rad = float(projected_waypoint.get("heading_rad", 0.0))
        return [
            float(projected_position[0]),
            float(projected_position[1]),
            0.0,
            float(projected_heading_rad),
        ]

    def update_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
        lane_center_waypoints: Sequence[Mapping[str, object]],
    ) -> List[float]:
        """
        Return the current final destination.

        Once the lane change trigger fires, the destination is projected onto
        the lane that contains the anchor point and then stays locked there.
        """

        _ = object_snapshots
        _ = float(simulation_time_s)

        current = list(current_final_destination_state) if len(current_final_destination_state) >= 2 else [0.0, 0.0, 0.0, 0.0]
        while len(current) < 4:
            current.append(0.0)

        if not self.enabled:
            return current

        if self._locked_destination is not None:
            return list(self._locked_destination)

        ego_x_m = float(ego_snapshot.get("x", 0.0))
        triggered_now = (ego_x_m >= self.trigger_x_m) if self.trigger_when_ego_x_ge else (ego_x_m <= self.trigger_x_m)
        if not triggered_now:
            return current

        anchor_waypoint = self._closest_waypoint(
            lane_center_waypoints=lane_center_waypoints,
            target_x_m=float(self.lane_change_anchor_xy[0]),
            target_y_m=float(self.lane_change_anchor_xy[1]),
        )
        if anchor_waypoint is None:
            return current

        target_lane_id = int(anchor_waypoint.get("lane_id", -1))
        projected_destination = self._project_destination_to_lane(
            current_final_destination_state=current,
            lane_center_waypoints=lane_center_waypoints,
            target_lane_id=target_lane_id,
        )
        if projected_destination is None:
            return current

        self._locked_destination = list(projected_destination)
        return list(self._locked_destination)
