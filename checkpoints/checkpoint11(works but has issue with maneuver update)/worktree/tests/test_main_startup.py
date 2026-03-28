import unittest

from main import _is_retriable_world_ready_error


class MainStartupTests(unittest.TestCase):
    def test_runtime_error_is_retriable(self):
        self.assertTrue(_is_retriable_world_ready_error(RuntimeError("world not ready")))

    def test_known_carla_color_overflow_value_error_is_retriable(self):
        self.assertTrue(
            _is_retriable_world_ready_error(
                ValueError("color: integer overflow in color channel")
            )
        )

    def test_other_value_error_is_not_retriable(self):
        self.assertFalse(_is_retriable_world_ready_error(ValueError("unexpected payload")))


if __name__ == "__main__":
    unittest.main()
