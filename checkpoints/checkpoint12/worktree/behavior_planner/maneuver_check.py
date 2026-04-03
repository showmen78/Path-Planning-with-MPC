"""
Maneuver feasibility checker.

Runs in a separate thread at 5-10 Hz.
For each drivable lane, determines whether the next maneuver (LEFT / RIGHT /
STRAIGHT) is reachable from that lane by walking CARLA waypoints forward
until a junction is found.

At the junction, outgoing branches are classified by yaw difference:
    delta_yaw > +20 deg  -> LEFT
    delta_yaw < -20 deg  -> RIGHT
    otherwise            -> STRAIGHT
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_STEP_M = 2.0
DEFAULT_MAX_WALK_M = 200.0
DEFAULT_FREQUENCY_HZ = 5.0
YAW_THRESHOLD_DEG = 20.0


# --------------------------------------------------------------------- #
# Helpers                                                                 #
# --------------------------------------------------------------------- #
def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize to [-180, 180]."""
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def _classify_branch(delta_yaw_deg: float) -> str:
    """Classify a junction branch by yaw difference."""
    d = _normalize_angle_deg(delta_yaw_deg)
    if d > YAW_THRESHOLD_DEG:
        return "LEFT"
    if d < -YAW_THRESHOLD_DEG:
        return "RIGHT"
    return "STRAIGHT"


# Route maneuver string -> branch classification
_MANEUVER_MAP = {
    "L": "LEFT", "LEFT": "LEFT", "TURN LEFT": "LEFT", "LEFT TURN": "LEFT",
    "R": "RIGHT", "RIGHT": "RIGHT", "TURN RIGHT": "RIGHT", "RIGHT TURN": "RIGHT",
    "S": "STRAIGHT", "STRAIGHT": "STRAIGHT", "CONTINUE STRAIGHT": "STRAIGHT",
}


# --------------------------------------------------------------------- #
# Checker (pure computation + CARLA waypoint API)                         #
# --------------------------------------------------------------------- #
class ManeuverChecker:
    """Check which maneuvers are reachable from each lane."""

    def __init__(
        self,
        step_m: float = DEFAULT_STEP_M,
        max_walk_m: float = DEFAULT_MAX_WALK_M,
    ) -> None:
        self._step_m = max(0.5, float(step_m))
        self._max_walk_m = max(10.0, float(max_walk_m))

    def check_maneuver_from_waypoint(
        self,
        start_wp: Any,
        required_maneuver: str,
    ) -> tuple[bool, float]:
        """
        Walk forward from *start_wp* using ``wp.next()`` until a junction.
        At the junction classify outgoing branches and return whether
        *required_maneuver* is available.

        Returns
        -------
        (maneuver_possible, distance_to_intersection_m)
        """
        wp = start_wp
        cumulative_m = 0.0

        # Walk forward until junction or max distance
        while cumulative_m < self._max_walk_m:
            if wp.is_junction:
                break
            candidates = wp.next(self._step_m)
            if not candidates:
                # Dead-end road — maneuver not possible
                return False, cumulative_m
            # Pick the straightest successor (smallest yaw change)
            current_yaw = wp.transform.rotation.yaw
            best = min(
                candidates,
                key=lambda c: abs(
                    _normalize_angle_deg(c.transform.rotation.yaw - current_yaw)
                ),
            )
            cumulative_m += self._step_m
            wp = best

        distance_to_intersection_m = cumulative_m

        if not wp.is_junction:
            # No junction within max_walk — any maneuver is fine (no constraint)
            return True, distance_to_intersection_m

        # At junction: get outgoing branches and classify
        branches = wp.next(self._step_m)
        if not branches:
            return False, distance_to_intersection_m

        junction_yaw = wp.transform.rotation.yaw
        available: set[str] = set()
        for branch in branches:
            delta = _normalize_angle_deg(
                branch.transform.rotation.yaw - junction_yaw
            )
            available.add(_classify_branch(delta))

        # Map the requested maneuver to our classification
        target = _MANEUVER_MAP.get(
            str(required_maneuver).strip().upper(), "STRAIGHT"
        )
        return target in available, distance_to_intersection_m

    def check_all_lanes(
        self,
        world_map: Any,
        carla: Any,
        ego_location: Any,
        ego_lane_id: int,
        available_lane_ids: Sequence[int],
        required_maneuver: str,
    ) -> tuple[Dict[int, bool], float]:
        """
        Check maneuver feasibility for every available lane.

        The method discovers lanes via CARLA ``get_left_lane`` /
        ``get_right_lane`` starting from the ego waypoint.  Results are
        keyed by internal lane-id (1 .. N, higher = more left).

        Returns
        -------
        (maneuver_possible_per_lane, min_distance_to_intersection_m)
        """
        ego_wp = world_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            # Cannot project to road — assume everything is possible
            return {lid: True for lid in available_lane_ids}, float("inf")

        result: Dict[int, bool] = {}
        min_dist = float("inf")

        # Ego lane (offset 0 -> ego_lane_id)
        possible, dist = self.check_maneuver_from_waypoint(ego_wp, required_maneuver)
        result[ego_lane_id] = possible
        min_dist = min(min_dist, dist)

        # Left lanes (get_left_lane -> higher internal lane_id)
        wp = ego_wp
        lid = ego_lane_id
        while True:
            left = wp.get_left_lane()
            if left is None:
                break
            if left.lane_type != carla.LaneType.Driving:
                break
            # Ensure same driving direction (CARLA lane_id sign)
            if left.lane_id * ego_wp.lane_id < 0:
                break
            lid += 1
            p, d = self.check_maneuver_from_waypoint(left, required_maneuver)
            result[lid] = p
            min_dist = min(min_dist, d)
            wp = left

        # Right lanes (get_right_lane -> lower internal lane_id)
        wp = ego_wp
        lid = ego_lane_id
        while True:
            right = wp.get_right_lane()
            if right is None:
                break
            if right.lane_type != carla.LaneType.Driving:
                break
            if right.lane_id * ego_wp.lane_id < 0:
                break
            lid -= 1
            p, d = self.check_maneuver_from_waypoint(right, required_maneuver)
            result[lid] = p
            min_dist = min(min_dist, d)
            wp = right

        return result, min_dist


# --------------------------------------------------------------------- #
# Background thread wrapper                                               #
# --------------------------------------------------------------------- #
class ManeuverCheckThread:
    """Continuously check maneuver feasibility in a daemon thread."""

    def __init__(
        self,
        checker: ManeuverChecker,
        frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    ) -> None:
        self._checker = checker
        self._period_s = 1.0 / max(0.5, float(frequency_hz))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Shared buffers
        self._input_data: Dict[str, Any] | None = None
        self._latest_result: Dict[str, Any] = {
            "maneuver_possible": {},
            "distance_to_intersection_m": float("inf"),
        }

    # -- main-loop interface ------------------------------------------- #
    def update_inputs(
        self,
        world_map: Any,
        carla: Any,
        ego_location: Any,
        ego_lane_id: int,
        available_lane_ids: Sequence[int],
        required_maneuver: str,
    ) -> None:
        """Push latest data from the main loop (non-blocking)."""
        with self._lock:
            self._input_data = {
                "world_map": world_map,
                "carla": carla,
                "ego_location": ego_location,
                "ego_lane_id": int(ego_lane_id),
                "available_lane_ids": list(available_lane_ids),
                "required_maneuver": str(required_maneuver),
            }

    def get_latest_result(self) -> Dict[str, Any]:
        """Read latest results (non-blocking)."""
        with self._lock:
            return {
                "maneuver_possible": dict(self._latest_result.get("maneuver_possible", {})),
                "distance_to_intersection_m": float(
                    self._latest_result.get("distance_to_intersection_m", float("inf"))
                ),
            }

    # -- lifecycle ----------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # -- worker -------------------------------------------------------- #
    def _worker(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                data = self._input_data

            if data is not None:
                try:
                    maneuver_map, dist = self._checker.check_all_lanes(
                        world_map=data["world_map"],
                        carla=data["carla"],
                        ego_location=data["ego_location"],
                        ego_lane_id=data["ego_lane_id"],
                        available_lane_ids=data["available_lane_ids"],
                        required_maneuver=data["required_maneuver"],
                    )
                    with self._lock:
                        self._latest_result = {
                            "maneuver_possible": maneuver_map,
                            "distance_to_intersection_m": float(dist),
                        }
                except Exception:
                    pass  # CARLA API errors should not kill the thread

            next_tick += self._period_s
            wait_s = max(0.0, next_tick - time.monotonic())
            self._stop_event.wait(wait_s)
