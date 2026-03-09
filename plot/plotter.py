"""
Plotting helpers for MPC_custom simulation outputs.

Outputs:
- Ego velocity vs longitudinal distance
- Ego acceleration vs longitudinal distance
- Ego steering angle vs longitudinal distance
- MPC cost terms vs longitudinal distance

Dimension rule used:
- Plot width [px] = pygame window width * plot_width_scale
- Plot height [px] = road total width in pixels * plot_height_scale
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import List, Mapping, Sequence
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


@dataclass
class PlotSizeSpec:
    """Container for required plot dimensions."""

    width_px: int
    height_px: int
    dpi: int = 100

    @property
    def figsize_in(self) -> tuple[float, float]:
        width_px = max(320, int(self.width_px))
        height_px = max(120, int(self.height_px))
        dpi = max(50, int(self.dpi))
        return (float(width_px) / float(dpi), float(height_px) / float(dpi))


class SimulationPlotter:
    """Generate and save simulation plots after the pygame run ends."""

    def __init__(
        self,
        output_dir: str,
        width_px: int,
        height_px: int,
        dpi: int = 100,
        trajectory_height_scale: float = 1.0,
    ) -> None:
        self.output_dir = str(output_dir)
        self.size = PlotSizeSpec(width_px=int(width_px), height_px=int(height_px), dpi=int(dpi))
        self.trajectory_height_scale = max(1.0, float(trajectory_height_scale))
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.size.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _new_figure(self) -> plt.Figure:
        fig = plt.figure(figsize=self.size.figsize_in, dpi=self.size.dpi)
        return fig

    def _new_trajectory_figure(self) -> plt.Figure:
        width_in, height_in = self.size.figsize_in
        fig = plt.figure(
            figsize=(width_in, max(height_in, height_in * float(self.trajectory_height_scale))),
            dpi=self.size.dpi,
        )
        return fig

    @staticmethod
    def _nice_step(value: float) -> float:
        value = max(1e-6, float(value))
        exponent = 10.0 ** int(__import__("math").floor(__import__("math").log10(value)))
        base = value / exponent
        if base <= 1.0:
            nice_base = 1.0
        elif base <= 2.0:
            nice_base = 2.0
        elif base <= 5.0:
            nice_base = 5.0
        else:
            nice_base = 10.0
        return float(nice_base * exponent)

    @staticmethod
    def _interpolate_y_at_query_x(
        src_x: Sequence[float],
        src_y: Sequence[float],
        query_x: Sequence[float],
    ) -> List[float]:
        """Linear interpolation y(query_x) for monotonic src_x."""

        if len(src_x) == 0 or len(src_y) == 0 or len(src_x) != len(src_y):
            return [0.0 for _ in query_x]
        x = [float(v) for v in src_x]
        y = [float(v) for v in src_y]
        q = [float(v) for v in query_x]

        out: List[float] = []
        x0 = x[0]
        xn = x[-1]

        for qv in q:
            if qv <= x0:
                out.append(float(y[0]))
                continue
            if qv >= xn:
                out.append(float(y[-1]))
                continue
            hi = bisect_right(x, qv)
            lo = max(0, hi - 1)
            hi = min(len(x) - 1, hi)
            xl = float(x[lo])
            xh = float(x[hi])
            yl = float(y[lo])
            yh = float(y[hi])
            if abs(xh - xl) <= 1e-12:
                out.append(float(yl))
                continue
            w = (qv - xl) / (xh - xl)
            out.append(float(yl + w * (yh - yl)))
        return out

    def _set_x_axis_from_world_x(self, ax: plt.Axes, x_values: Sequence[float]) -> None:
        if len(x_values) == 0:
            return
        x_min = min(float(v) for v in x_values)
        x_max = max(float(v) for v in x_values)
        if abs(x_max - x_min) < 1e-9:
            x_min -= 1.0
            x_max += 1.0
        ax.set_xlim(x_min, x_max)
        step = self._nice_step((x_max - x_min) / 10.0)
        ax.xaxis.set_major_locator(MultipleLocator(step))

    def save_ego_timeseries_plots(
        self,
        scenario_name: str,
        time_s: Sequence[float],
        x_m: Sequence[float],
        y_m: Sequence[float],
        velocity_mps: Sequence[float],
        accel_mps2: Sequence[float],
        steer_rad: Sequence[float],
    ) -> List[str]:
        """Save control/state history plots for ego vehicle."""

        outputs: List[str] = []
        if len(time_s) == 0:
            return outputs

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{scenario_name}_{timestamp}"

        fig = self._new_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_s, velocity_mps, color="#2ca02c", linewidth=2.0)
        ax.set_title("Ego Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]")
        ax.grid(True, alpha=0.3)
        outputs.append(self._save(fig, f"{prefix}_ego_velocity.png"))

        fig = self._new_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_s, accel_mps2, color="#ff7f0e", linewidth=2.0)
        ax.set_title("Ego Acceleration")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("acceleration [m/s^2]")
        ax.grid(True, alpha=0.3)
        outputs.append(self._save(fig, f"{prefix}_ego_acceleration.png"))

        fig = self._new_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_s, steer_rad, color="#9467bd", linewidth=2.0)
        ax.set_title("Ego Steering Angle")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("steering [rad]")
        ax.grid(True, alpha=0.3)
        outputs.append(self._save(fig, f"{prefix}_ego_steering.png"))

        return outputs

    def save_cost_plot(
        self,
        scenario_name: str,
        time_s: Sequence[float],
        cost_terms: Mapping[str, Sequence[float]],
    ) -> List[str]:
        """Save MPC cost-term plot for active terms."""

        outputs: List[str] = []
        if len(time_s) == 0:
            return outputs

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{scenario_name}_{timestamp}"

        fig = self._new_figure()
        ax = fig.add_subplot(1, 1, 1)

        ordered_keys = [
            "Cost_ref",
            "Cost_lane_keeping",
            "Cost_LaneCenter",
            "Cost_Repulsive_Safe",
            "Cost_Repulsive_Collision",
            "Cost_Control",
        ]
        colors = {
            "Cost_ref": "#1f77b4",
            "Cost_lane_keeping": "#17becf",
            "Cost_LaneCenter": "#17becf",
            "Cost_Repulsive_Safe": "#bcbd22",
            "Cost_Repulsive_Collision": "#d62728",
            "Cost_Control": "#ff7f0e",
        }
        labels = {
            "Cost_ref": "J_ref",
            "Cost_lane_keeping": "J_lane_keeping",
            "Cost_LaneCenter": "J_lane_keeping",
            "Cost_Repulsive_Safe": "J_repulsive_safe",
            "Cost_Repulsive_Collision": "J_repulsive_collision",
            "Cost_Control": "J_control",
        }

        for key in ordered_keys:
            values = list(cost_terms.get(key, []))
            if len(values) != len(time_s):
                continue
            label = labels.get(key, key)
            ax.plot(time_s, values, label=label, linewidth=1.8, color=colors.get(key, None))

        ax.set_title("MPC Cost Terms")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        outputs.append(self._save(fig, f"{prefix}_mpc_cost_terms.png"))
        return outputs

    def save_x_coordinate_plots(
        self,
        scenario_name: str,
        ego_time_s: Sequence[float],
        x_m: Sequence[float],
        y_m: Sequence[float],
        velocity_mps: Sequence[float],
        accel_mps2: Sequence[float],
        steer_rad: Sequence[float],
        cost_time_s: Sequence[float],
        cost_terms: Mapping[str, Sequence[float]],
        include_cost: bool = True,
        include_properties: bool = True,
    ) -> List[str]:
        """
        Save longitudinal-distance-based plots for scenario analysis.

        X-axis for these figures is the world longitudinal coordinate x [m].
        """

        outputs: List[str] = []
        if len(x_m) == 0 or len(ego_time_s) == 0:
            return outputs
        if not bool(include_cost) and not bool(include_properties):
            return outputs

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{scenario_name}_{timestamp}"

        x_series = [float(v) for v in x_m]

        if bool(include_properties):
            fig = self._new_figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x_series, velocity_mps, color="#2ca02c", linewidth=2.0)
            ax.set_title("Ego Velocity vs Longitudinal Distance")
            ax.set_xlabel("longitudinal distance x [m]")
            ax.set_ylabel("velocity [m/s]")
            self._set_x_axis_from_world_x(ax, x_series)
            ax.grid(True, alpha=0.3)
            outputs.append(self._save(fig, f"{prefix}_ego_velocity_vs_x.png"))

            fig = self._new_figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x_series, accel_mps2, color="#ff7f0e", linewidth=2.0)
            ax.set_title("Ego Acceleration vs Longitudinal Distance")
            ax.set_xlabel("longitudinal distance x [m]")
            ax.set_ylabel("acceleration [m/s^2]")
            self._set_x_axis_from_world_x(ax, x_series)
            ax.grid(True, alpha=0.3)
            outputs.append(self._save(fig, f"{prefix}_ego_acceleration_vs_x.png"))

            fig = self._new_figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x_series, steer_rad, color="#9467bd", linewidth=2.0)
            ax.set_title("Ego Steering Angle vs Longitudinal Distance")
            ax.set_xlabel("longitudinal distance x [m]")
            ax.set_ylabel("steering [rad]")
            self._set_x_axis_from_world_x(ax, x_series)
            ax.grid(True, alpha=0.3)
            outputs.append(self._save(fig, f"{prefix}_ego_steering_vs_x.png"))

        if bool(include_cost) and len(cost_time_s) > 0:
            cost_x = self._interpolate_y_at_query_x(src_x=ego_time_s, src_y=x_series, query_x=cost_time_s)

            fig = self._new_figure()
            ax = fig.add_subplot(1, 1, 1)
            ordered_keys = [
                "Cost_ref",
                "Cost_lane_keeping",
                "Cost_LaneCenter",
                "Cost_Repulsive_Safe",
                "Cost_Repulsive_Collision",
                "Cost_Control",
            ]
            colors = {
                "Cost_ref": "#1f77b4",
                "Cost_lane_keeping": "#17becf",
                "Cost_LaneCenter": "#17becf",
                "Cost_Repulsive_Safe": "#bcbd22",
                "Cost_Repulsive_Collision": "#d62728",
                "Cost_Control": "#ff7f0e",
            }
            labels = {
                "Cost_ref": "J_ref",
                "Cost_lane_keeping": "J_lane_keeping",
                "Cost_LaneCenter": "J_lane_keeping",
                "Cost_Repulsive_Safe": "J_repulsive_safe",
                "Cost_Repulsive_Collision": "J_repulsive_collision",
                "Cost_Control": "J_control",
            }
            for key in ordered_keys:
                values = list(cost_terms.get(key, []))
                if len(values) != len(cost_time_s):
                    continue
                label = labels.get(key, key)
                ax.plot(cost_x, values, label=label, linewidth=1.8, color=colors.get(key, None))

            ax.set_title("MPC Cost Terms vs Longitudinal Distance")
            ax.set_xlabel("longitudinal distance x [m]")
            ax.set_ylabel("cost")
            self._set_x_axis_from_world_x(ax, cost_x)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            outputs.append(self._save(fig, f"{prefix}_mpc_cost_terms_vs_x.png"))

        return outputs

    def save_mpc_plan_step_plots(
        self,
        scenario_name: str,
        replan_x_m: Sequence[float],
        accel_by_step: Sequence[Sequence[float]],
        steer_by_step: Sequence[Sequence[float]],
        velocity_by_step: Sequence[Sequence[float]],
        psi_by_step: Sequence[Sequence[float]],
    ) -> List[str]:
        """
        Save MPC planned state/control step profiles versus ego longitudinal x.

        Each input series contains one line per MPC step index, sampled once per
        replan. Example: accel_by_step[0] is the planned first-step acceleration
        across all replans.
        """

        outputs: List[str] = []
        if len(replan_x_m) == 0:
            return outputs

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{scenario_name}_{timestamp}"
        x_series = [float(v) for v in replan_x_m]

        def _plot_multistep_series(
            filename_suffix: str,
            title: str,
            ylabel: str,
            value_by_step: Sequence[Sequence[float]],
            color_cycle: Sequence[str],
        ) -> None:
            has_data = any(len(series) == len(x_series) for series in value_by_step)
            if not has_data:
                return

            fig = self._new_figure()
            ax = fig.add_subplot(1, 1, 1)
            for step_idx, series in enumerate(value_by_step):
                values = list(series)
                if len(values) != len(x_series):
                    continue
                ax.plot(
                    x_series,
                    values,
                    linewidth=1.8,
                    color=color_cycle[step_idx % len(color_cycle)],
                    label=f"step {step_idx}",
                )
            ax.set_title(title)
            ax.set_xlabel("ego longitudinal distance x at replan [m]")
            ax.set_ylabel(ylabel)
            self._set_x_axis_from_world_x(ax, x_series)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            outputs.append(self._save(fig, f"{prefix}_{filename_suffix}.png"))

        _plot_multistep_series(
            filename_suffix="mpc_planned_acceleration_steps_vs_x",
            title="MPC Planned Acceleration Steps vs Longitudinal Distance",
            ylabel="acceleration [m/s^2]",
            value_by_step=accel_by_step,
            color_cycle=["#ff7f0e", "#d95f02", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"],
        )
        _plot_multistep_series(
            filename_suffix="mpc_planned_steering_steps_vs_x",
            title="MPC Planned Steering Steps vs Longitudinal Distance",
            ylabel="steering [rad]",
            value_by_step=steer_by_step,
            color_cycle=["#9467bd", "#6a3d9a", "#8073ac", "#9e9ac8", "#bcbddc", "#dadaeb"],
        )
        _plot_multistep_series(
            filename_suffix="mpc_planned_velocity_steps_vs_x",
            title="MPC Planned Velocity Steps vs Longitudinal Distance",
            ylabel="velocity [m/s]",
            value_by_step=velocity_by_step,
            color_cycle=["#2ca02c", "#238b45", "#41ab5d", "#74c476", "#a1d99b", "#c7e9c0"],
        )
        _plot_multistep_series(
            filename_suffix="mpc_planned_heading_steps_vs_x",
            title="MPC Planned Heading Steps vs Longitudinal Distance",
            ylabel="heading psi [rad]",
            value_by_step=psi_by_step,
            color_cycle=["#1f77b4", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"],
        )

        return outputs
