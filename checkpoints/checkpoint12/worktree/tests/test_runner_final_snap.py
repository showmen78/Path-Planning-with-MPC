import unittest

from carla_scenario.runner import _apply_final_destination_snap


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

    def test_snaps_blue_dot_to_final_destination_and_sets_stop_speed(self):
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
        self.assertAlmostEqual(float(temporary_destination_state[2]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[3]), 1.2)
        self.assertAlmostEqual(active_v_max, 7.0 * (2.0 / 5.0))


if __name__ == "__main__":
    unittest.main()
