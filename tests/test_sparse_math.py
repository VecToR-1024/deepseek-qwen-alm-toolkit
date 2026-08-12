import unittest

from topk_distill.sparse_math import sparse_forward_kl


class SparseForwardKlTests(unittest.TestCase):
    def test_is_zero_for_matching_sparse_distributions(self):
        loss = sparse_forward_kl(
            teacher_logprobs=[-0.2, -2.0],
            student_logprobs=[-0.2, -2.0],
            add_tail=True,
        )

        self.assertAlmostEqual(loss, 0.0, places=12)

    def test_tail_bucket_prevents_top1_from_collapsing_to_zero(self):
        with_tail = sparse_forward_kl(
            teacher_logprobs=[-0.1],
            student_logprobs=[-1.0],
            add_tail=True,
        )
        renormalized = sparse_forward_kl(
            teacher_logprobs=[-0.1],
            student_logprobs=[-1.0],
            add_tail=False,
        )

        self.assertGreater(with_tail, 0.0)
        self.assertAlmostEqual(renormalized, 0.0, places=12)

    def test_rejects_logprob_mass_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "probability mass"):
            sparse_forward_kl(
                teacher_logprobs=[-0.1, -0.1],
                student_logprobs=[-1.0, -1.0],
                add_tail=True,
            )


if __name__ == "__main__":
    unittest.main()
