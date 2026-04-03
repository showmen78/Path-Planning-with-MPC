import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from carla_scenario import list_available_scenarios, load_carla_scenario
from carla_scenario.high_level_route_planning.scenario import (
    filter_dynamic_obstacle_snapshots,
    initialize_runtime,
    maybe_replan_global_route,
    spawn_obstacles,
)


class _FakeCityObjectLabel:
    Any = "Any"


class _FakeCarla:
    CityObjectLabel = _FakeCityObjectLabel

    class LaneType:
        Driving = "Driving"

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch = float(pitch)
            self.yaw = float(yaw)
            self.roll = float(roll)

    class Transform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    class VehicleControl:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class Vector3D:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)


class _FakeEnvironmentObject:
    def __init__(self, name: str, x: float, y: float, z: float = 0.0):
        self.name = str(name)
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=float(z))
        )


class _FakeActor:
    def __init__(self, *, type_id: str = "", role_name: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.type_id = str(type_id)
        self.attributes = {"role_name": str(role_name)} if role_name else {}
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        )

    def get_transform(self):
        return self._transform


class _FakeSpawnVehicle:
    def __init__(self, transform, role_name: str = ""):
        self._transform = transform
        self.attributes = {"role_name": str(role_name)} if role_name else {}

    def set_simulate_physics(self, _enabled):
        return None

    def set_autopilot(self, _enabled):
        return None

    def set_target_velocity(self, _velocity):
        return None

    def set_target_angular_velocity(self, _velocity):
        return None

    def apply_control(self, _control):
        return None

    def get_transform(self):
        return self._transform


class _FakeBlueprint:
    def __init__(self):
        self.attributes = {}

    def has_attribute(self, _name: str) -> bool:
        return True

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[str(name)] = str(value)


class _FakeBlueprintLibrary:
    def __init__(self):
        self.last_blueprint = None

    def find(self, _blueprint_id: str):
        self.last_blueprint = _FakeBlueprint()
        return self.last_blueprint


class _FakeWorld:
    def __init__(self, *, environment_objects=None, actors=None):
        self._environment_objects = list(environment_objects or [])
        self._actors = list(actors or [])

    def get_environment_objects(self, _label):
        return list(self._environment_objects)

    def get_actors(self):
        return list(self._actors)


class _FakeSpawnWorld(_FakeWorld):
    def __init__(self, *, environment_objects=None, actors=None):
        super().__init__(environment_objects=environment_objects, actors=actors)
        self.spawn_calls = []

    def try_spawn_actor(self, blueprint, transform):
        self.spawn_calls.append((blueprint, transform))
        return _FakeSpawnVehicle(
            transform=transform,
            role_name=str(getattr(blueprint, "attributes", {}).get("role_name", "")),
        )


class _FakeWaypointWorldMap:
    def __init__(self, waypoint_xy=None, road_id=12, section_id=0, lane_id=1):
        self._waypoint_xy = list(waypoint_xy or [0.0, 0.0])
        self._road_id = int(road_id)
        self._section_id = int(section_id)
        self._lane_id = int(lane_id)

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del location
        del project_to_road
        del lane_type
        return SimpleNamespace(
            road_id=int(self._road_id),
            section_id=int(self._section_id),
            lane_id=int(self._lane_id),
            get_left_lane=lambda: None,
            get_right_lane=lambda: None,
            transform=SimpleNamespace(
                location=SimpleNamespace(
                    x=float(self._waypoint_xy[0]),
                    y=float(self._waypoint_xy[1]),
                    z=0.0,
                ),
                rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
            )
        )


class HighLevelRoutePlanningScenarioTests(unittest.TestCase):
    def test_scenario_is_available_and_uses_expected_runtime_module(self):
        self.assertIn("high_level_route_planning", list_available_scenarios())

        scenario_cfg = load_carla_scenario("high_level_route_planning")

        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.high_level_route_planning.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.high_level_route_planning.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("workzone_name", "")),
            "workzone",
        )
        self.assertIn(
            "traffic_obstacle1",
            list(scenario_cfg.get("obstacles", {}).get("marker_names", [])),
        )

    def test_workzone_message_is_inserted_once_after_delay(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "workzone_name": "workzone",
                        "cooperative_message_delay_s": 5.0,
                        "cp_message_id": "workzone_message",
                        "cp_message_path": message_path,
                    }
                },
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                wall_time_s=100.0,
            )

            route_summary, route_points, next_runtime_state = maybe_replan_global_route(
                runtime_state=runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=4.9,
                wall_time_s=104.9,
            )
            self.assertIsNone(route_summary)
            self.assertIsNone(route_points)
            self.assertFalse(bool(next_runtime_state.get("cp_message_inserted", False)))

            with open(message_path, "r", encoding="utf-8") as message_file:
                self.assertEqual(json.load(message_file), [])

            route_summary, route_points, next_runtime_state = maybe_replan_global_route(
                runtime_state=next_runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=5.0,
                wall_time_s=105.0,
            )
            self.assertIsNone(route_summary)
            self.assertIsNone(route_points)
            self.assertTrue(bool(next_runtime_state.get("cp_message_inserted", False)))

            with open(message_path, "r", encoding="utf-8") as message_file:
                messages = json.load(message_file)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["id"], "workzone_message")
            self.assertEqual(messages[0]["type"], "lane_closure")
            self.assertEqual(messages[0]["position"], [13.0, 35.0])
            self.assertEqual(messages[0]["road_id"], 20)
            self.assertEqual(messages[0]["section_id"], 0)
            self.assertEqual(messages[0]["lane_id"], 1)

            _, _, final_runtime_state = maybe_replan_global_route(
                runtime_state=next_runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=6.0,
                wall_time_s=106.0,
            )
            self.assertTrue(bool(final_runtime_state.get("cp_message_inserted", False)))

            with open(message_path, "r", encoding="utf-8") as message_file:
                repeated_messages = json.load(message_file)
            self.assertEqual(len(repeated_messages), 1)

    def test_initialize_runtime_warns_when_workzone_is_missing(self):
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "workzone_name": "workzone",
                    }
                },
                world=_FakeWorld(),
                world_map=_FakeWaypointWorldMap([1.0, 2.0]),
                carla=_FakeCarla,
            )

        self.assertIsNone(runtime_state.get("workzone_position_xy", None))
        self.assertIn("Warning: workzone object was not found.", captured_stdout.getvalue())
        self.assertIn("workzone", captured_stdout.getvalue())

    def test_initialize_runtime_does_not_fallback_to_actor_when_workzone_cube_is_missing(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "workzone_name": "workzone",
                },
            },
            world=_FakeWorld(
                actors=[
                    _FakeActor(role_name="workzone", x=12.5, y=34.0),
                ],
            ),
            world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
            carla=_FakeCarla,
        )

        self.assertIsNone(runtime_state.get("workzone_object_name"))
        self.assertIsNone(runtime_state.get("workzone_position_xy"))
        self.assertIsNone(runtime_state.get("workzone_road_id"))
        self.assertIsNone(runtime_state.get("workzone_lane_id"))

    def test_spawn_obstacles_falls_back_to_actor_marker_when_environment_marker_is_missing(self):
        world = _FakeSpawnWorld(
            actors=[
                _FakeActor(role_name="traffic_obstacle1", x=12.5, y=34.0, z=0.0),
            ],
        )
        blueprint_library = _FakeBlueprintLibrary()

        spawned = spawn_obstacles(
            world=world,
            world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
            carla=_FakeCarla,
            blueprint_library=blueprint_library,
            scenario_cfg={
                "obstacles": {
                    "marker_names": ["traffic_obstacle1"],
                    "vehicle_blueprint": "vehicle.tesla.model3",
                    "color_rgb": "90,90,90",
                    "spawn_z_offset_m": 0.05,
                }
            },
        )

        self.assertEqual(len(spawned), 1)
        self.assertEqual(len(world.spawn_calls), 1)
        self.assertEqual(
            str(blueprint_library.last_blueprint.attributes.get("role_name", "")),
            "traffic_obstacle1",
        )
        spawned_transform = world.spawn_calls[0][1]
        self.assertAlmostEqual(float(spawned_transform.location.x), 12.5, places=3)
        self.assertAlmostEqual(float(spawned_transform.location.y), 34.0, places=3)
        self.assertAlmostEqual(float(spawned_transform.location.z), 0.05, places=3)

    def test_workzone_is_retried_and_inserted_if_found_later(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "workzone_name": "workzone",
                        "cooperative_message_delay_s": 5.0,
                        "cp_message_id": "workzone_retry_message",
                        "cp_message_path": message_path,
                    }
                },
                world=_FakeWorld(),
                world_map=_FakeWaypointWorldMap([0.0, 0.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                wall_time_s=200.0,
            )

            _, _, next_runtime_state = maybe_replan_global_route(
                runtime_state=runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=40.0, y=50.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([41.0, 51.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=5.0,
                wall_time_s=205.0,
            )

            self.assertTrue(bool(next_runtime_state.get("cp_message_inserted", False)))
            with open(message_path, "r", encoding="utf-8") as message_file:
                messages = json.load(message_file)
            self.assertEqual(messages[0]["position"], [41.0, 51.0])
            self.assertEqual(messages[0]["road_id"], 20)
            self.assertEqual(messages[0]["section_id"], 0)
            self.assertEqual(messages[0]["lane_id"], 1)

    def test_workzone_does_not_inject_virtual_obstacle_after_delay(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "workzone_name": "workzone",
                    "cooperative_message_delay_s": 5.0,
                }
            },
            world=_FakeWorld(
                environment_objects=[
                    _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                ]
            ),
            world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
            carla=_FakeCarla,
            wall_time_s=300.0,
        )

        snapshots_before, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            world=_FakeWorld(
                environment_objects=[
                    _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                ]
            ),
            world_map=_FakeWaypointWorldMap([13.0, 35.0]),
            carla=_FakeCarla,
            object_snapshots=[],
            sim_time_s=4.0,
            wall_time_s=304.0,
        )
        self.assertEqual(snapshots_before, [])

        snapshots_after, _ = filter_dynamic_obstacle_snapshots(
            runtime_state=next_runtime_state,
            world=_FakeWorld(
                environment_objects=[
                    _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                ]
            ),
            world_map=_FakeWaypointWorldMap([13.0, 35.0]),
            carla=_FakeCarla,
            object_snapshots=[],
            sim_time_s=5.0,
            wall_time_s=305.0,
        )
        self.assertEqual(snapshots_after, [])

    def test_cooperative_message_delay_uses_wall_time_not_sim_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "workzone_name": "workzone",
                        "cooperative_message_delay_s": 5.0,
                        "cp_message_id": "workzone_wall_time_message",
                        "cp_message_path": message_path,
                    }
                },
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                wall_time_s=400.0,
            )

            _, _, next_runtime_state = maybe_replan_global_route(
                runtime_state=runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=999.0,
                wall_time_s=404.9,
            )
            self.assertFalse(bool(next_runtime_state.get("cp_message_inserted", False)))

            _, _, final_runtime_state = maybe_replan_global_route(
                runtime_state=next_runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("workzone", x=12.5, y=34.0),
                    ]
                ),
                world_map=_FakeWaypointWorldMap([13.0, 35.0], road_id=20, section_id=0, lane_id=-2),
                carla=_FakeCarla,
                sim_time_s=999.0,
                wall_time_s=405.0,
            )
            self.assertTrue(bool(final_runtime_state.get("cp_message_inserted", False)))


if __name__ == "__main__":
    unittest.main()
