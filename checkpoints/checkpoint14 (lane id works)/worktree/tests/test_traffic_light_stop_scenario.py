import contextlib
import io
import unittest
from types import SimpleNamespace

from carla_scenario import list_available_scenarios, load_carla_scenario
from carla_scenario.traffic_light_stop.scenario import (
    filter_dynamic_obstacle_snapshots,
    initialize_runtime,
    spawn_obstacles,
)


class _FakeCityObjectLabel:
    Any = "Any"


class _FakeTrafficLightState:
    Green = "Green"
    Yellow = "Yellow"
    Red = "Red"


class _FakeCarla:
    CityObjectLabel = _FakeCityObjectLabel
    TrafficLightState = _FakeTrafficLightState

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
            location=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        )


class _FakeActor:
    def __init__(
        self,
        *,
        type_id: str = "",
        role_name: str = "",
        name: str = "",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ):
        self.type_id = str(type_id)
        self.attributes = {}
        if role_name:
            self.attributes["role_name"] = str(role_name)
        if name:
            self.attributes["name"] = str(name)
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        )

    def get_transform(self):
        return self._transform


class _FakeTrafficLightActor(_FakeActor):
    def __init__(self, *, name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        super().__init__(
            type_id="traffic.traffic_light",
            name=str(name),
            x=float(x),
            y=float(y),
            z=float(z),
        )
        self.last_state = None
        self.frozen = False

    def set_state(self, state):
        self.last_state = state

    def freeze(self, enabled: bool):
        self.frozen = bool(enabled)


class _FakeWorld:
    def __init__(self, *, environment_objects=None, actors=None):
        self._environment_objects = list(environment_objects or [])
        self._actors = list(actors or [])

    def get_environment_objects(self, _label):
        return list(self._environment_objects)

    def get_actors(self):
        return list(self._actors)


class _FakeSpawnVehicle:
    def __init__(self, transform, role_name: str = ""):
        self._transform = transform
        self.attributes = {"role_name": str(role_name)} if role_name else {}
        self.target_velocity = None
        self.target_angular_velocity = None
        self.autopilot_enabled = None

    def set_simulate_physics(self, _enabled):
        return None

    def set_autopilot(self, enabled):
        self.autopilot_enabled = bool(enabled)

    def set_target_velocity(self, velocity):
        self.target_velocity = velocity

    def set_target_angular_velocity(self, velocity):
        self.target_angular_velocity = velocity

    def apply_control(self, control):
        self.control = control

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


class _FakeSpawnWorld(_FakeWorld):
    def __init__(self, *, environment_objects=None, actors=None):
        super().__init__(environment_objects=environment_objects, actors=actors)
        self.spawn_calls = []

    def try_spawn_actor(self, blueprint, transform):
        vehicle = _FakeSpawnVehicle(
            transform=transform,
            role_name=str(getattr(blueprint, "attributes", {}).get("role_name", "")),
        )
        self.spawn_calls.append((blueprint, transform, vehicle))
        return vehicle


class _FakeWaypointWorldMap:
    def __init__(self, *, x: float, y: float, yaw_deg: float):
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=SimpleNamespace(pitch=0.0, yaw=float(yaw_deg), roll=0.0),
        )

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del location, project_to_road, lane_type
        return SimpleNamespace(transform=self._transform)


class TrafficLightStopScenarioTests(unittest.TestCase):
    def test_scenario_is_available_and_uses_expected_runtime_module(self):
        self.assertIn("traffic_light_stop", list_available_scenarios())

        scenario_cfg = load_carla_scenario("traffic_light_stop")

        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.traffic_light_stop.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("ego_spawn", "")),
            "traffic_ego",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("trigger_yellow_marker", "")),
            "trigger_yellow",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("trigger_red_marker", "")),
            "trigger_red",
        )
        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.traffic_light_stop.scenario",
        )
        self.assertIn(
            "traffic_obstacle1",
            list(scenario_cfg.get("obstacles", {}).get("marker_names", [])),
        )

    def test_initialize_runtime_warns_when_trigger_markers_are_missing(self):
        traffic_light = _FakeTrafficLightActor(
            name="BP_TrafficLightNew_T10_master_largeBIG_rsc11",
        )
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "traffic_light_name": "BP_TrafficLightNew_T10_master_largeBIG_rsc11",
                        "trigger_yellow_marker": "trigger_yellow",
                        "trigger_red_marker": "trigger_red",
                    }
                },
                world=_FakeWorld(actors=[traffic_light]),
                carla=_FakeCarla,
            )

        self.assertIsNone(runtime_state.get("trigger_yellow_transform"))
        self.assertIsNone(runtime_state.get("trigger_red_transform"))
        output = captured_stdout.getvalue()
        self.assertIn("trigger marker 'trigger_yellow' was not found", output)
        self.assertIn("trigger marker 'trigger_red' was not found", output)

    def test_initialize_runtime_finds_signal_and_triggers_via_partial_name_match(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject(
                    "SM_trigger_yellow_001",
                    x=10.0,
                    y=0.0,
                ),
                _FakeEnvironmentObject(
                    "SM_trigger_red_001",
                    x=20.0,
                    y=0.0,
                ),
                _FakeEnvironmentObject(
                    "BP_TrafficLightNew_T10_master_largeBIG_rsc11_component",
                    x=1.0,
                    y=2.0,
                ),
            ],
            actors=[
                _FakeTrafficLightActor(
                    name="traffic_light_runtime_actor",
                    x=1.5,
                    y=2.0,
                ),
            ],
        )

        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "traffic_light_name": "BP_TrafficLightNew_T10_master_largeBIG_rsc11",
                    "trigger_yellow_marker": "trigger_yellow",
                    "trigger_red_marker": "trigger_red",
                }
            },
            world=world,
            carla=_FakeCarla,
        )

        self.assertIsNotNone(runtime_state.get("traffic_light_actor"))
        self.assertIsNotNone(runtime_state.get("trigger_yellow_transform"))
        self.assertIsNotNone(runtime_state.get("trigger_red_transform"))

    def test_spawn_obstacles_uses_exact_marker_transform_resolution(self):
        world = _FakeSpawnWorld(
            environment_objects=[
                _FakeEnvironmentObject("SM_traffic_obstacle1_001", x=12.5, y=34.0),
            ],
        )
        blueprint_library = _FakeBlueprintLibrary()

        spawned = spawn_obstacles(
            world=world,
            world_map=_FakeWaypointWorldMap(x=20.0, y=50.0, yaw_deg=37.0),
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
        spawned_transform = world.spawn_calls[0][1]
        spawned_vehicle = world.spawn_calls[0][2]
        self.assertAlmostEqual(float(spawned_transform.location.x), 12.5, places=3)
        self.assertAlmostEqual(float(spawned_transform.location.y), 34.0, places=3)
        self.assertAlmostEqual(float(spawned_transform.location.z), 0.05, places=3)
        self.assertAlmostEqual(float(spawned_transform.rotation.yaw), 37.0, places=3)
        self.assertEqual(
            str(blueprint_library.last_blueprint.attributes.get("role_name", "")),
            "traffic_obstacle1",
        )
        self.assertFalse(bool(spawned_vehicle.autopilot_enabled))
        self.assertEqual(float(spawned_vehicle.target_velocity.x), 0.0)
        self.assertEqual(float(spawned_vehicle.target_velocity.y), 0.0)
        self.assertEqual(float(spawned_vehicle.target_velocity.z), 0.0)

    def test_signal_transitions_green_to_yellow_to_red_then_back_to_green(self):
        traffic_light_name = "BP_TrafficLightNew_T10_master_largeBIG_rsc11"
        traffic_light = _FakeTrafficLightActor(name=traffic_light_name, x=0.0, y=0.0)
        ego_actor = _FakeActor(role_name="ego", x=0.0, y=0.0)
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject("trigger_yellow", x=10.0, y=0.0),
                _FakeEnvironmentObject("trigger_red", x=20.0, y=0.0),
            ],
            actors=[traffic_light, ego_actor],
        )

        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "traffic_light_name": traffic_light_name,
                    "trigger_yellow_marker": "trigger_yellow",
                    "trigger_red_marker": "trigger_red",
                    "trigger_distance_m": 2.0,
                    "red_hold_duration_s": 5.0,
                }
            },
            world=world,
            carla=_FakeCarla,
            wall_time_s=100.0,
        )
        self.assertEqual(traffic_light.last_state, _FakeTrafficLightState.Green)
        self.assertTrue(bool(traffic_light.frozen))
        self.assertEqual(str(runtime_state.get("phase")), "green_initial")

        ego_actor._transform.location.x = 9.2
        filtered_snapshots, runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            world=world,
            carla=_FakeCarla,
            object_snapshots=[{"vehicle_id": "obs1"}],
            wall_time_s=101.0,
        )
        self.assertEqual(filtered_snapshots, [{"vehicle_id": "obs1"}])
        self.assertEqual(traffic_light.last_state, _FakeTrafficLightState.Yellow)
        self.assertEqual(str(runtime_state.get("phase")), "yellow")

        ego_actor._transform.location.x = 19.2
        filtered_snapshots, runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            world=world,
            carla=_FakeCarla,
            object_snapshots=[],
            wall_time_s=102.0,
        )
        self.assertEqual(filtered_snapshots, [])
        self.assertEqual(traffic_light.last_state, _FakeTrafficLightState.Red)
        self.assertEqual(str(runtime_state.get("phase")), "red")
        self.assertAlmostEqual(
            float(runtime_state.get("red_release_wall_time_s")),
            107.0,
            places=3,
        )

        _, runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            world=world,
            carla=_FakeCarla,
            object_snapshots=[],
            wall_time_s=106.9,
        )
        self.assertEqual(traffic_light.last_state, _FakeTrafficLightState.Red)
        self.assertEqual(str(runtime_state.get("phase")), "red")

        _, runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            world=world,
            carla=_FakeCarla,
            object_snapshots=[],
            wall_time_s=107.0,
        )
        self.assertEqual(traffic_light.last_state, _FakeTrafficLightState.Green)
        self.assertEqual(str(runtime_state.get("phase")), "green_released")


if __name__ == "__main__":
    unittest.main()
