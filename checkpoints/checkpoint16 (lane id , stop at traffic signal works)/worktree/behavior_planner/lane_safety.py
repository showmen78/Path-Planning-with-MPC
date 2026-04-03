"""
Lane safety scoring module.

For each drivable lane (same direction), computes a safety score in [0, 1]
using distance, time-to-collision (TTC), and TTC trend (linear regression).

Safety terms per obstacle:
    phi_d   = max(d - d_safe, 0) / (max(d - d_safe, 0) + d_safe)
    phi_ttc = max(TTC - TTC_safe, 0) / (max(TTC - TTC_safe, 0) + TTC_safe)
    phi_m   = 1 / (1 + exp(-k * m))          (m = TTC slope from regression)
    S_obs   = phi_d * phi_ttc * phi_m

Hard constraint: if d < d_safe OR TTC < TTC_safe then S_obs = 0.

Lane score:  S_lane = min(S_front, S_rear)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_D_SAFE_M = 8.0
DEFAULT_TTC_SAFE_S = 3.0
DEFAULT_SIGMOID_K = 5.0
DEFAULT_TTC_HISTORY_SIZE = 10
_EPS = 1e-6


# --------------------------------------------------------------------- #
# Scorer (pure computation, no threading)                                 #
# --------------------------------------------------------------------- #
class LaneSafetyScorer:
    """Compute per-lane safety scores from obstacle data."""

    def __init__(
        self,
        d_safe_m: float = DEFAULT_D_SAFE_M,
        ttc_safe_s: float = DEFAULT_TTC_SAFE_S,
        sigmoid_k: float = DEFAULT_SIGMOID_K,
        ttc_history_size: int = DEFAULT_TTC_HISTORY_SIZE,
    ) -> None:
        self._d_safe_m = float(d_safe_m)
        self._ttc_safe_s = float(ttc_safe_s)
        self._sigmoid_k = float(sigmoid_k)
        self._ttc_history_size = int(ttc_history_size)
        # obstacle_id -> deque of (timestamp_s, ttc_s)
        self._ttc_history: Dict[str, deque] = {}

    # ----------------------------------------------------------------- #
    # TTC helpers                                                         #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _compute_ttc(distance_m: float, delta_v_mps: float) -> float:
        """TTC = d / max(eps, dv).  If dv <= 0 the obstacle is not approaching."""
        if delta_v_mps <= 0.0:
            return float("inf")
        return float(distance_m) / max(_EPS, float(delta_v_mps))

    def _update_ttc_history(self, obstacle_id: str, ttc_s: float, timestamp_s: float) -> None:
        if obstacle_id not in self._ttc_history:
            self._ttc_history[obstacle_id] = deque(maxlen=self._ttc_history_size)
        self._ttc_history[obstacle_id].append((float(timestamp_s), float(ttc_s)))

    def _compute_ttc_slope(self, obstacle_id: str) -> float:
        """Linear-regression slope of TTC vs time.  Positive = improving."""
        history = self._ttc_history.get(obstacle_id)
        if history is None or len(history) < 3:
            return 0.0

        times = np.array([e[0] for e in history], dtype=np.float64)
        ttcs = np.array([e[1] for e in history], dtype=np.float64)
        # Cap infinite TTCs for numerical stability
        ttcs = np.clip(ttcs, 0.0, 100.0)

        t_mean = np.mean(times)
        ttc_mean = np.mean(ttcs)
        numerator = float(np.sum((times - t_mean) * (ttcs - ttc_mean)))
        denominator = float(np.sum((times - t_mean) ** 2))
        if abs(denominator) < _EPS:
            return 0.0
        return numerator / denominator

    # ----------------------------------------------------------------- #
    # Single-obstacle score                                               #
    # ----------------------------------------------------------------- #
    def _obstacle_score(self, distance_m: float, ttc_s: float, ttc_slope: float) -> float:
        """S_obs in [0, 1]."""
        d_safe = self._d_safe_m
        ttc_safe = self._ttc_safe_s
        k = self._sigmoid_k

        # Hard constraint
        if distance_m < d_safe or ttc_s < ttc_safe:
            return 0.0

        # Soft terms
        d_excess = max(distance_m - d_safe, 0.0)
        phi_d = d_excess / (d_excess + d_safe)

        ttc_excess = max(ttc_s - ttc_safe, 0.0)
        phi_ttc = ttc_excess / (ttc_excess + ttc_safe)

        # Sigmoid on slope: positive slope (TTC growing) -> safer
        phi_m = 1.0 / (1.0 + math.exp(-k * ttc_slope))

        return float(phi_d * phi_ttc * phi_m)

    # ----------------------------------------------------------------- #
    # Per-lane scoring                                                    #
    # ----------------------------------------------------------------- #
    def compute_lane_scores(
        self,
        ego_snapshot: Mapping[str, float],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        ego_lane_id: int,
        available_lane_ids: Sequence[int],
        timestamp_s: float,
    ) -> Dict[int, float]:
        """
        Compute safety score for every available lane.

        Parameters
        ----------
        ego_snapshot        : {x, y, v, psi}
        obstacle_snapshots  : list of {vehicle_id, x, y, v, psi, ...}
        lane_assignments    : vehicle_id -> internal lane_id
        ego_lane_id         : ego's current internal lane_id
        available_lane_ids  : [1, 2, ..., N]
        timestamp_s         : simulation clock (for TTC history)

        Returns
        -------
        {lane_id: score}  where score in [0, 1].
        """
        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_v = float(ego_snapshot.get("v", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)

        # Group obstacles by lane
        lane_obstacles: Dict[int, List[Mapping[str, Any]]] = {
            lid: [] for lid in available_lane_ids
        }
        for obs in obstacle_snapshots:
            obs_id = str(obs.get("vehicle_id", ""))
            obs_lane = lane_assignments.get(obs_id, -1)
            if obs_lane in lane_obstacles:
                lane_obstacles[obs_lane].append(obs)

        scores: Dict[int, float] = {}

        for lane_id in available_lane_ids:
            obstacles = lane_obstacles.get(lane_id, [])
            if len(obstacles) == 0:
                scores[lane_id] = 1.0
                continue

            # Find nearest front and nearest rear obstacle (longitudinal)
            nearest_front = None
            nearest_rear = None
            min_front_d = float("inf")
            min_rear_d = float("inf")

            for obs in obstacles:
                dx = float(obs.get("x", 0.0)) - ego_x
                dy = float(obs.get("y", 0.0)) - ego_y
                longitudinal = cos_h * dx + sin_h * dy

                if longitudinal >= 0.0:
                    if longitudinal < min_front_d:
                        min_front_d = longitudinal
                        nearest_front = obs
                else:
                    rear_d = abs(longitudinal)
                    if rear_d < min_rear_d:
                        min_rear_d = rear_d
                        nearest_rear = obs

            # Front obstacle score
            s_front = 1.0
            if nearest_front is not None:
                obs_v = float(nearest_front.get("v", 0.0))
                delta_v = ego_v - obs_v  # positive = ego closing in
                ttc = self._compute_ttc(min_front_d, delta_v)
                obs_id = str(nearest_front.get("vehicle_id", f"_front_{lane_id}"))
                self._update_ttc_history(obs_id, ttc, timestamp_s)
                slope = self._compute_ttc_slope(obs_id)
                s_front = self._obstacle_score(min_front_d, ttc, slope)

            # Rear obstacle score
            s_rear = 1.0
            if nearest_rear is not None:
                obs_v = float(nearest_rear.get("v", 0.0))
                delta_v = obs_v - ego_v  # positive = rear closing in
                ttc = self._compute_ttc(min_rear_d, delta_v)
                obs_id = str(nearest_rear.get("vehicle_id", f"_rear_{lane_id}"))
                self._update_ttc_history(obs_id, ttc, timestamp_s)
                slope = self._compute_ttc_slope(obs_id)
                s_rear = self._obstacle_score(min_rear_d, ttc, slope)

            scores[lane_id] = min(s_front, s_rear)

        return scores

    def cleanup_stale_obstacles(self, active_ids: set[str]) -> None:
        """Remove TTC history for obstacles that disappeared."""
        stale = [oid for oid in self._ttc_history if oid not in active_ids]
        for oid in stale:
            del self._ttc_history[oid]
