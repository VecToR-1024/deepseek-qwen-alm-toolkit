import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trl.experimental.distillation import DistillationTrainer

from topk_distill.trainer import PluggableDistillationTrainer


class _FakeClient:
    supports_actual_logprobs = False


def _fake_base_init(self, *args, **kwargs):
    self.args = kwargs["args"]
    self.teacher_client = None
    self.teacher_model = None
    self.use_teacher_server = False


class PluggableDistillationTrainerTests(unittest.TestCase):
    @patch.object(DistillationTrainer, "__init__", _fake_base_init)
    def test_injects_custom_client_without_starting_the_builtin_vllm_client(self):
        args = SimpleNamespace(beta=0.0, loss_top_k=8, use_teacher_server=False)
        client = _FakeClient()

        trainer = PluggableDistillationTrainer(args=args, teacher_client=client)

        self.assertIs(trainer.teacher_client, client)
        self.assertTrue(trainer.use_teacher_server)
        self.assertTrue(trainer.args.use_teacher_server)

    @patch.object(DistillationTrainer, "__init__", _fake_base_init)
    def test_rejects_reverse_kl_when_client_cannot_score_actual_tokens(self):
        args = SimpleNamespace(beta=1.0, loss_top_k=1, use_teacher_server=False)

        with self.assertRaisesRegex(ValueError, "actual-token logprobs"):
            PluggableDistillationTrainer(args=args, teacher_client=_FakeClient())


if __name__ == "__main__":
    unittest.main()
