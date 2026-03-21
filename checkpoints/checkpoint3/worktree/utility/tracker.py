"""
Obstacle trajectory tracking and prediction module for MPC_custom.

This file defines a `Tracker` class that:
1. Stores recent state history for each non-ego vehicle.
2. Fits time-parameterized polynomials x(t), y(t) at each timestep.
3. Projects obstacle future trajectories over a short horizon (default 3 s).
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Sequence
import math
import warnings

import numpy as np


@dataclass
class TrackerConfig:
    """
    Intent:
        Store tracker hyperparameters used for obstacle trajectory fitting.

    Fields:
        prediction_horizon_s:
            float [s], how far in the future obstacle trajectories are projected.
        polynomial_degree:
            int scalar, requested polynomial degree for x(t) and y(t).
            The effective degree is auto-limited by available data points.
        fit_window_s:
            float [s], only samples newer than `now - fit_window_s` are used.
        max_history_points:
            int scalar, ring-buffer cap for samples per obstacle.
        min_points_for_polyfit:
            int scalar, minimum points before using polynomial fit.
            Below this count, constant/linear fallback is used.
    """

    prediction_horizon_s: float
    polynomial_degree: int
    fit_window_s: float
    max_history_points: int
    min_points_for_polyfit: int


class Tracker:
    """
    Intent:
        Track obstacle motion and predict future obstacle trajectories.

    Logic and Method:
        - At each timestep, the tracker ingests current obstacle snapshots.
        - For each obstacle ID, it stores time-stamped history:
            [timestamp_s, x, y, v, psi].
        - To predict, it fits independent least-squares polynomials:
            x(t) and y(t), where t is relative time from latest sample.
        - It evaluates these polynomials for future times t = dt, 2*dt, ..., H,
          and also derives velocity/heading from polynomial derivatives.

    Inputs / Outputs:
        - `update(obstacle_snapshots, timestamp_s)`:
            obstacle_snapshots: sequence[dict], each containing at least:
                vehicle_id: str
                x, y, v, psi: float
            timestamp_s: float [s], current simulation timestamp.
        - `predict(step_dt_s, horizon_s=None)`:
            step_dt_s: float [s], sampling interval of requested prediction.
            horizon_s: float [s] | None, optional override of horizon.
            Returns:
                dict[str, list[dict[str, float]]]
                key: obstacle vehicle_id
                value: predicted sequence with fields:
                    t_s, x, y, v, psi
                shape per obstacle: (N, 5), N = round(horizon / step_dt_s).
    """

    def __init__(self, tracker_cfg: Mapping[str, object] | None = None) -> None:
        """
        Intent:
            Initialize tracker configuration and internal history buffers.

        Inputs:
            tracker_cfg:
                optional mapping with keys:
                    prediction_horizon_s, polynomial_degree, fit_window_s,
                    max_history_points, min_points_for_polyfit.
        """

        cfg = dict(tracker_cfg or {})
        self.config = TrackerConfig(
            prediction_horizon_s=max(0.1, float(cfg.get("prediction_horizon_s", 3.0))),
            polynomial_degree=max(1, int(cfg.get("polynomial_degree", 2))),
            fit_window_s=max(0.2, float(cfg.get("fit_window_s", 4.0))),
            max_history_points=max(8, int(cfg.get("max_history_points", 240))),
            min_points_for_polyfit=max(2, int(cfg.get("min_points_for_polyfit", 3))),
        )

        # obstacle_id -> deque of historical samples.
        # Each sample dict stores:
        #   timestamp_s, x, y, v, psi
        self._history_by_obstacle_id: Dict[str, Deque[Dict[str, float]]] = {}
        self._last_timestamp_s: float | None = None

    def update(self, obstacle_snapshots: Sequence[Mapping[str, object]], timestamp_s: float) -> None:
        """
        Intent:
            Ingest current obstacle states and refresh per-obstacle histories.

        Logic:
            1. Append one timestamped sample per obstacle.
            2. Keep only active obstacle IDs from this frame.
            3. Remove stale samples older than fit_window_s.

        Inputs:
            obstacle_snapshots:
                sequence[Mapping[str, object]] with fields:
                    vehicle_id: str
                    x, y, v, psi: float
            timestamp_s:
                float [s], current simulation time.

        Output:
            None.
        """

        now_s = float(timestamp_s)
        self._last_timestamp_s = now_s

        active_ids: set[str] = set()
        min_keep_time_s = now_s - self.config.fit_window_s

        for snapshot in obstacle_snapshots:
            obstacle_id = str(snapshot.get("vehicle_id", ""))
            if obstacle_id == "":
                continue

            active_ids.add(obstacle_id)
            if obstacle_id not in self._history_by_obstacle_id:
                self._history_by_obstacle_id[obstacle_id] = deque(maxlen=self.config.max_history_points)

            sample = {
                "timestamp_s": now_s,
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "v": float(snapshot.get("v", 0.0)),
                "psi": float(snapshot.get("psi", 0.0)),
            }
            self._history_by_obstacle_id[obstacle_id].append(sample)

        # Drop removed obstacles.
        stale_ids = set(self._history_by_obstacle_id.keys()) - active_ids
        for stale_id in stale_ids:
            self._history_by_obstacle_id.pop(stale_id, None)

        # Window old samples out of active histories.
        for obstacle_id, history_deque in self._history_by_obstacle_id.items():
            while len(history_deque) > 0 and float(history_deque[0]["timestamp_s"]) < min_keep_time_s:
                history_deque.popleft()

    @staticmethod
    def _fit_position_polynomial(
        time_rel_s: np.ndarray,
        values: np.ndarray,
        requested_degree: int,
    ) -> np.ndarray:
        """
        Intent:
            Fit one polynomial value(t) using least squares.

        Inputs:
            time_rel_s:
                np.ndarray, shape (M,), relative timestamps.
            values:
                np.ndarray, shape (M,), samples corresponding to time_rel_s.
            requested_degree:
                int scalar, desired polynomial degree.

        Output:
            np.ndarray polynomial coefficients (highest order first),
            shape (degree + 1,).
        """

        point_count = int(values.shape[0])
        degree = int(max(1, min(requested_degree, point_count - 1)))
        return np.polyfit(time_rel_s, values, deg=degree)

    @staticmethod
    def _fallback_predict(
        latest_sample: Mapping[str, float],
        future_time_s: float,
    ) -> Dict[str, float]:
        """
        Intent:
            Predict one future sample with constant velocity + heading fallback.

        Inputs:
            latest_sample:
                mapping with x, y, v, psi values at current time.
            future_time_s:
                float [s], prediction time from current sample.

        Output:
            dict[str, float] with fields t_s, x, y, v, psi.
        """

        x0 = float(latest_sample["x"])
        y0 = float(latest_sample["y"])
        v0 = float(latest_sample["v"])
        psi0 = float(latest_sample["psi"])

        x_pred = x0 + v0 * math.cos(psi0) * future_time_s
        y_pred = y0 + v0 * math.sin(psi0) * future_time_s
        return {"t_s": future_time_s, "x": x_pred, "y": y_pred, "v": v0, "psi": psi0}

    def predict(
        self,
        step_dt_s: float,
        horizon_s: float | None = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Intent:
            Generate future obstacle trajectories from tracked history.

        Method:
            For each obstacle:
            - If insufficient samples, use constant-velocity fallback.
            - Else fit x(t), y(t) polynomials and evaluate future samples.
            - Derive v, psi from first derivatives dx/dt, dy/dt.

        Inputs:
            step_dt_s:
                float [s], temporal spacing between predicted points.
            horizon_s:
                optional float [s], prediction horizon override.

        Output:
            dict[str, list[dict[str, float]]], predicted trajectory per obstacle.
        """

        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be > 0.")

        horizon_value_s = float(self.config.prediction_horizon_s if horizon_s is None else horizon_s)
        horizon_value_s = max(step_dt_s, horizon_value_s)
        steps = max(1, int(round(horizon_value_s / step_dt_s)))

        prediction_by_obstacle_id: Dict[str, List[Dict[str, float]]] = {}

        for obstacle_id, history_deque in self._history_by_obstacle_id.items():
            history = list(history_deque)
            if len(history) == 0:
                continue

            latest = history[-1]
            prediction: List[Dict[str, float]] = []

            if len(history) < self.config.min_points_for_polyfit:
                for step_idx in range(steps):
                    t_future_s = (step_idx + 1) * step_dt_s
                    prediction.append(self._fallback_predict(latest, t_future_s))
                prediction_by_obstacle_id[obstacle_id] = prediction
                continue

            t0 = float(latest["timestamp_s"])
            time_rel_s = np.array([float(sample["timestamp_s"]) - t0 for sample in history], dtype=float)
            x_values = np.array([float(sample["x"]) for sample in history], dtype=float)
            y_values = np.array([float(sample["y"]) for sample in history], dtype=float)

            # Polyfit can emit RankWarning for near-singular matrices when all
            # timestamps or positions are almost constant; fallback handles that.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coeff_x = self._fit_position_polynomial(
                        time_rel_s=time_rel_s,
                        values=x_values,
                        requested_degree=self.config.polynomial_degree,
                    )
                    coeff_y = self._fit_position_polynomial(
                        time_rel_s=time_rel_s,
                        values=y_values,
                        requested_degree=self.config.polynomial_degree,
                    )
                dcoeff_x = np.polyder(coeff_x)
                dcoeff_y = np.polyder(coeff_y)

                for step_idx in range(steps):
                    t_future_s = (step_idx + 1) * step_dt_s
                    x_pred = float(np.polyval(coeff_x, t_future_s))
                    y_pred = float(np.polyval(coeff_y, t_future_s))
                    vx_pred = float(np.polyval(dcoeff_x, t_future_s))
                    vy_pred = float(np.polyval(dcoeff_y, t_future_s))
                    v_pred = float(math.hypot(vx_pred, vy_pred))

                    if v_pred > 1e-5:
                        psi_pred = float(math.atan2(vy_pred, vx_pred))
                    else:
                        psi_pred = float(latest["psi"])

                    prediction.append(
                        {
                            "t_s": t_future_s,
                            "x": x_pred,
                            "y": y_pred,
                            "v": v_pred,
                            "psi": psi_pred,
                        }
                    )
            except Exception:
                prediction = []
                for step_idx in range(steps):
                    t_future_s = (step_idx + 1) * step_dt_s
                    prediction.append(self._fallback_predict(latest, t_future_s))

            prediction_by_obstacle_id[obstacle_id] = prediction

        return prediction_by_obstacle_id

    def get_histories(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Intent:
            Return deep-copied obstacle histories for debugging/inspection.

        Output:
            dict[str, list[dict[str, float]]], obstacle history map.
        """

        return {key: deepcopy(list(value)) for key, value in self._history_by_obstacle_id.items()}
