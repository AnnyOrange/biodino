import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rolling_1pb_eval_queue.py"
SPEC = importlib.util.spec_from_file_location("rolling_1pb_eval_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class DenseCapacityTests(unittest.TestCase):
    def test_two_dense_tasks_fit_on_40_gib_card(self):
        self.assertTrue(queue.dense_capacity(14909, 40960, 1, 16000, 2))

    def test_two_dense_tasks_do_not_fit_on_24_gib_card(self):
        self.assertFalse(queue.dense_capacity(3000, 24576, 1, 16000, 2))

    def test_actual_usage_and_task_limit_block_admission(self):
        self.assertFalse(queue.dense_capacity(25000, 40960, 1, 16000, 2))
        self.assertFalse(queue.dense_capacity(2000, 40960, 2, 16000, 2))


if __name__ == "__main__":
    unittest.main()
