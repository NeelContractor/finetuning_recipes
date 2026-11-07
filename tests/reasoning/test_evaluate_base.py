import unittest

from reasoning.env import score_rollouts, summarize


class RewardModel:
    def batch_score(self, responses, references, batch_size=8):
        return [0.8 for _ in responses]


class EvaluateBaseTests(unittest.TestCase):
    def test_score_rollouts_matches_reward_space(self):
        results = score_rollouts(
            [{
                "reference": "plain answer",
                "completion": "<think>reason carefully</think> plain answer",
            }],
            RewardModel(),
        )

        expected = {
            "think_format_reward": 1.0,
            "output_datatype_reward": 0.5,
            "output_schema_reward": 0.5,
            "doom_loop_reward": 0.0,
            "neuraltxt_reward": 0.8,
            "length_penalty_reward": 0.0,
        }
        self.assertEqual(set(results[0]["rewards"]), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(results[0]["rewards"][name], value, places=5)
        # 1.0 + 0.5 + 0.5 + 0.0 + 0.8 + 0.5 * 0.0 = 2.8
        self.assertAlmostEqual(results[0]["total_reward"], 2.8, places=5)

    def test_sft_mode_skips_format_rewards(self):
        results = score_rollouts(
            [{
                "reference": "plain answer",
                "completion": "plain answer",
            }],
            RewardModel(),
            require_think_format=False,
            include_format_rewards=False,
        )

        rewards = results[0]["rewards"]
        self.assertEqual(rewards["think_format_reward"], 0.0)
        self.assertEqual(rewards["output_datatype_reward"], 0.0)
        self.assertEqual(rewards["output_schema_reward"], 0.0)
        self.assertAlmostEqual(rewards["neuraltxt_reward"], 0.8, places=5)
        self.assertAlmostEqual(results[0]["total_reward"], 0.8, places=5)

    def test_hard_cap_is_a_flat_penalty(self):
        reference = " ".join(f"w{i}" for i in range(10))
        completion = "<think>reason</think> " + " ".join(f"w{i}" for i in range(60))

        results = score_rollouts(
            [{"reference": reference, "completion": completion}],
            RewardModel(),
        )

        self.assertEqual(results[0]["rewards"]["length_penalty_reward"], -1.0)
        # 1.0 think + 0.5 datatype + 0.5 schema + 0.8 neuraltxt
        # + 0.5 * (-1.0 length) = 2.3
        self.assertAlmostEqual(results[0]["total_reward"], 2.3, places=5)

    def test_summary_reports_failure_rates(self):
        results = score_rollouts(
            [
                {
                    "reference": "plain answer",
                    "completion": "<think>x</think> plain answer",
                },
                {
                    "reference": "plain answer",
                    "completion": '<think>x x x x</think> {"answer": 1}',
                },
            ],
            RewardModel(),
        )
        summary = summarize(results)

        self.assertEqual(summary["num_rollouts"], 2)
        self.assertEqual(summary["think_format_pass_rate"], 1.0)
        self.assertEqual(summary["datatype_match_rate"], 0.5)
        self.assertEqual(summary["schema_match_rate"], 0.5)
        self.assertEqual(summary["doom_loop_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
