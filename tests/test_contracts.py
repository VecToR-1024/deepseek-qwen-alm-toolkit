import math
import unittest

from topk_distill.contracts import normalize_sequence_logprobs


class NormalizeSequenceLogprobsTests(unittest.TestCase):
    def test_pads_short_topk_rows_and_sorts_by_logprob(self):
        response = {
            "logprobs": [[[-2.0, -0.2], [-0.5]]],
            "logprob_token_ids": [[[12, 11], [13]]],
        }

        normalized = normalize_sequence_logprobs(
            response,
            completion_lengths=[2],
            top_k=3,
        )

        self.assertEqual(normalized["logprob_token_ids"], [[[11, 12, 0], [13, 0, 0]]])
        self.assertEqual(normalized["logprobs"][0][0][:2], [-0.2, -2.0])
        self.assertTrue(math.isinf(normalized["logprobs"][0][0][2]))
        self.assertEqual(normalized["actual_logprobs"], [[[-math.inf], [-math.inf]]])

    def test_rejects_a_completion_length_mismatch(self):
        response = {
            "logprobs": [[[-0.1]]],
            "logprob_token_ids": [[[7]]],
        }

        with self.assertRaisesRegex(ValueError, "completion positions"):
            normalize_sequence_logprobs(
                response,
                completion_lengths=[2],
                top_k=1,
            )

    def test_rejects_duplicate_token_ids_at_one_position(self):
        response = {
            "logprobs": [[[-0.1, -0.2]]],
            "logprob_token_ids": [[[7, 7]]],
        }

        with self.assertRaisesRegex(ValueError, "duplicate token ids"):
            normalize_sequence_logprobs(
                response,
                completion_lengths=[1],
                top_k=2,
            )

    def test_preserves_actual_token_logprobs_when_the_api_returns_them(self):
        response = {
            "logprobs": [[[-0.1]]],
            "logprob_token_ids": [[[7]]],
            "actual_logprobs": [[[-1.25]]],
        }

        normalized = normalize_sequence_logprobs(
            response,
            completion_lengths=[1],
            top_k=1,
        )

        self.assertEqual(normalized["actual_logprobs"], [[[-1.25]]])


if __name__ == "__main__":
    unittest.main()
