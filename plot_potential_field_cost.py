"""
Standalone plot for the live repulsive-potential cost function.

This script visualizes the current project cost:
    J_safe      = w_s * exp(-k_s * (r_s - s_s))
    J_collision = w_c * exp(-k_c * (r_c - s_c))
    J_total     = J_safe + J_collision

Outputs:
    plot/potential_field_cost_function.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utility import load_yaml_file


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "plot" / "potential_field_cost_function.png"
MPC_YAML_PATH = PROJECT_ROOT / "MPC" / "mpc.yaml"


def main() -> None:
    cfg = load_yaml_file(str(MPC_YAML_PATH))
    repulsive_cfg = dict(cfg.get("mpc", {}).get("cost", {}).get("repulsive_potential", {}))

    w_safe = float(repulsive_cfg.get("w_safe_zone", 100.0))
    w_collision = float(repulsive_cfg.get("w_collision_zone", 100.0))
    k_safe = float(repulsive_cfg.get("safe_exponential_gain", 7.0))
    k_collision = float(repulsive_cfg.get("collision_exponential_gain", 10.0))
    s_safe = float(repulsive_cfg.get("safe_distance_shift", 1.5))
    s_collision = float(repulsive_cfg.get("collision_distance_shift", 1.5))

    r_s = np.linspace(0.0, 3.0, 500)
    r_c = np.linspace(0.0, 3.0, 500)

    j_safe = w_safe * np.exp(-k_safe * (r_s - s_safe))
    j_collision = w_collision * np.exp(-k_collision * (r_c - s_collision))

    rs_grid, rc_grid = np.meshgrid(r_s, r_c)
    j_total_grid = (
        w_safe * np.exp(-k_safe * (rs_grid - s_safe))
        + w_collision * np.exp(-k_collision * (rc_grid - s_collision))
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)

    ax = axes[0]
    ax.plot(r_s, j_safe, color="#bcbd22", linewidth=2.0)
    ax.axvline(s_safe, color="#555555", linestyle="--", linewidth=1.2, label=r"$s_s$")
    ax.set_title(r"$J_{\mathrm{safe}}(r_s)$")
    ax.set_xlabel(r"$r_s$")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(r_c, j_collision, color="#d62728", linewidth=2.0)
    ax.axvline(s_collision, color="#555555", linestyle="--", linewidth=1.2, label=r"$s_c$")
    ax.set_title(r"$J_{\mathrm{collision}}(r_c)$")
    ax.set_xlabel(r"$r_c$")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    contour = ax.contourf(rs_grid, rc_grid, j_total_grid, levels=30, cmap="viridis")
    ax.set_title(r"$J_{\mathrm{total}}(r_s, r_c)$")
    ax.set_xlabel(r"$r_s$")
    ax.set_ylabel(r"$r_c$")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("cost")

    fig.suptitle(
        r"Live Cost: $J = w_s e^{-k_s(r_s-s_s)} + w_c e^{-k_c(r_c-s_c)}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
