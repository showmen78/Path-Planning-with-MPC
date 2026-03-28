import unittest

from carla_scenario.runner import (
    _split_projected_polyline_segments,
    _split_route_world_segments,
)


class RunnerOverlayTests(unittest.TestCase):
    def test_projected_polyline_segments_break_on_missing_points(self):
        segments = _split_projected_polyline_segments(
            [(10, 10), (20, 20), None, (100, 100), (120, 120)]
        )

        self.assertEqual(
            segments,
            [
                [(10, 10), (20, 20)],
                [(100, 100), (120, 120)],
            ],
        )

    def test_route_world_segments_break_on_large_gaps(self):
        segments = _split_route_world_segments(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [30.0, 0.0],
                [31.0, 0.0],
            ],
            max_gap_m=5.0,
        )

        self.assertEqual(
            segments,
            [
                [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
                [(30.0, 0.0), (31.0, 0.0)],
            ],
        )


if __name__ == "__main__":
    unittest.main()
