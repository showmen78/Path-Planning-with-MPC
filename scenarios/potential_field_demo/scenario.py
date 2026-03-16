"""
Potential-field demo scenario.

Intent:
    Provide a minimal straight-road setup where the ego stays fixed and one
    obstacle approaches from behind in the same lane. This is used to inspect
    how the repulsive-potential costs change with ego-obstacle distance.
"""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Dict, List, Mapping, Sequence, Tuple

from road import RoadModel
from utility import load_yaml_file


class PotentialFieldDemoScenario:
    """Straight-road scenario used for potential-field cost visualization."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        self._scenario_yaml_path = str(scenario_yaml_path or default_yaml_path)
        self._config: Dict[str, object] = load_yaml_file(self._scenario_yaml_path)
        self._road_model = RoadModel()

    def get_config(self) -> Dict[str, object]:
        return deepcopy(self._config)

    def draw_road(
        self,
        surface,
        road_cfg: Mapping[str, object],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        world_to_screen_fn=None,
    ) -> None:
        _ = world_to_screen_fn
        self._road_model.draw(
            surface=surface,
            road_cfg=road_cfg,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
        )

    def get_latest_lane_waypoints(self) -> List[Dict[str, object]]:
        return self._road_model.get_latest_lane_waypoints()

    def get_ego_control_override(
        self,
        ego_snapshot: Mapping[str, object],
        destination_state: Sequence[float],
        final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> Mapping[str, bool]:
        """
        Keep the demo ego fixed in place.

        The planner still runs so the repulsive costs are evaluated and logged,
        but the executed ego motion is frozen to isolate the cost-vs-distance
        effect of the approaching obstacle.
        """

        _ = (ego_snapshot, destination_state, final_destination_state, simulation_time_s)
        return {"freeze": True}


def create_scenario() -> PotentialFieldDemoScenario:
    return PotentialFieldDemoScenario()
