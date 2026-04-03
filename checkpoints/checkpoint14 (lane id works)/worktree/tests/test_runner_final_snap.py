import unittest

from carla_scenario.runner import (
    _apply_final_destination_snap,
    _apply_stop_target_speed_cap,
)


class RunnerFinalDestinationSnapTests(unittest.TestCase):
    def test_keeps_temporary_destination_when_not_within_snap_distance(self):
        temporary_destination_state, active_v_max = _apply_final_destination_snap(
            temporary_destination_state=[0.0, 0.0, 5.0, 0.0, 1],
            final_destination_state=[10.0, 0.0, 0.0, 0.5],
            ego_state=[0.0, 0.0, 4.0, 0.0],
            lock_to_final_distance_m=5.0,
            original_max_velocity_mps=7.0,
        )

        self.assertEqual(temporary_destination_state, [0.0, 0.0, 5.0, 0.0, 1])
        self.assertAlmostEqual(active_v_max, 7.0)

    def test_snaps_blue_dot_to_final_destination_and_tapers_speed(self):
        temporary_destination_state, active_v_max = _apply_final_destination_snap(
            temporary_destination_state=[9.5, 0.2, 5.0, 0.0, 2, 0.0],
            final_destination_state=[10.0, 0.0, 0.0, 1.2],
            ego_state=[8.0, 0.0, 4.0, 0.0],
            lock_to_final_distance_m=5.0,
            original_max_velocity_mps=7.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertAlmostEqual(float(temporary_destination_state[0]), 10.0)
        self.assertAlmostEqual(float(temporary_destination_state[1]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), 7.0 * (2.0 / 5.0))
        self.assertAlmostEqual(float(temporary_destination_state[3]), 1.2)
        self.assertAlmostEqual(active_v_max, 7.0 * (2.0 / 5.0))

    def test_stop_target_speed_cap_shapes_speed_toward_stop_point_without_zeroing_immediately(self):
        temporary_destination_state, active_v_max = _apply_stop_target_speed_cap(
            temporary_destination_state=[0.0, 20.0, 0.0, 1.57, 1],
            ego_state=[0.0, 0.0, 6.0, 1.57],
            stop_target_distance_m=20.0,
            original_max_velocity_mps=7.0,
            braking_deceleration_mps2=4.0,
            stop_buffer_m=2.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertGreater(float(temporary_destination_state[2]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), float(active_v_max))
        self.assertLessEqual(float(active_v_max), 7.0)


if __name__ == "__main__":
    unittest.main()
