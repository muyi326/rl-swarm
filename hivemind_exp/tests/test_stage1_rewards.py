import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from hivemind_exp.gsm8k.stage1_rewards import (
    extract_xml_answer,
    count_xml,
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    top_k_cumulative_reward,
    hivemind_cumulative_reward,
)
from hivemind_exp.hivemind_utils import HivemindNode


class TestStage1Rewards(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_text_good = "<think>\nLet me solve this step by step.\n</think>\n<answer>\n42\n</answer>\n"
        self.sample_text_bad = "The answer is 42"
        self.sample_text_partial = "<think>\nLet me solve this.\n</think>\n<answer>\n42"

        # Mock data for reward functions
        self.mock_prompts = [[{"content": "What is 6 times 7?"}]]
        self.mock_completions = [
            [{"content": self.sample_text_good}],
            [{"content": self.sample_text_bad}],
            [{"content": self.sample_text_partial}],
        ]
        self.mock_answer = ["42"]

    def test_extract_xml_answer(self):
        """Test the extract_xml_answer function"""
        self.assertEqual(extract_xml_answer(self.sample_text_good), "42")
        self.assertEqual(extract_xml_answer("<answer>123</answer>"), "123")
        self.assertEqual(
            extract_xml_answer("Some text <answer>456</answer> more text"), "456"
        )
        # Edge cases
        self.assertEqual(extract_xml_answer("<answer></answer>"), "")
        # todo: When no answer tag is present, the implementation will return the string itself
        # as text.split("</answer>")[0] will be the original string, this behavior is probably wrong and should be corrected
        self.assertEqual(extract_xml_answer("No answer tag"), "No answer tag")

    def test_count_xml(self):
        """Test the count_xml function"""
        # Good format should get high score
        good_score = count_xml(self.sample_text_good)
        self.assertGreater(good_score, 0.4)
        # Bad format should get low score
        bad_score = count_xml(self.sample_text_bad)
        self.assertEqual(bad_score, 0.0)
        # Partial format should get partial score
        partial_score = count_xml(self.sample_text_partial)
        self.assertGreater(partial_score, bad_score)
        self.assertLess(partial_score, good_score)

        # Test with extra text after </answer>
        text_with_extra = self.sample_text_good + "extra text"
        self.assertLess(count_xml(text_with_extra), good_score)

    def test_correctness_reward_func(self):
        """Test the correctness_reward_func function"""
        # Use a single completion at a time as the function expects
        rewards1 = correctness_reward_func(
            self.mock_prompts,
            [self.mock_completions[0]],
            self.mock_answer,
            weighting=2.0,
        )
        rewards2 = correctness_reward_func(
            self.mock_prompts,
            [self.mock_completions[1]],
            self.mock_answer,
            weighting=2.0,
        )
        rewards3 = correctness_reward_func(
            self.mock_prompts,
            [self.mock_completions[2]],
            self.mock_answer,
            weighting=2.0,
        )

        # First completion has correct answer
        self.assertEqual(rewards1[0], 2.0)
        # Second completion doesn't have answer in XML format
        self.assertEqual(rewards2[0], 0.0)
        # Third completion has correct answer but incomplete format
        self.assertEqual(rewards3[0], 2.0)

    def test_int_reward_func(self):
        """Test the int_reward_func function"""
        # Create completions with different answer types
        completions = [
            [{"content": "<answer>\n42\n</answer>"}],
            [{"content": "<answer>\nforty-two\n</answer>"}],
            [{"content": "<answer>\n42.0\n</answer>"}],
        ]
        rewards = int_reward_func(completions, weighting=0.5)
        # First is an integer
        self.assertEqual(rewards[0], 0.5)
        # Second is not an integer
        self.assertEqual(rewards[1], 0.0)
        # Third is not an integer
        self.assertEqual(rewards[2], 0.0)

    def test_strict_format_reward_func(self):
        """Test the strict_format_reward_func function"""
        completions = [
            [{"content": self.sample_text_good}],
            [{"content": self.sample_text_bad}],
            [{"content": self.sample_text_partial}],
        ]
        rewards = strict_format_reward_func(completions, weighting=0.5)
        # First has correct format
        self.assertEqual(rewards[0], 0.5)
        # Second has incorrect format
        self.assertEqual(rewards[1], 0.0)
        # Third has incomplete format
        self.assertEqual(rewards[2], 0.0)

    def test_soft_format_reward_func(self):
        """Test the soft_format_reward_func function"""
        # Test with more lenient format requirements
        completions = [
            [{"content": self.sample_text_good}],
            [{"content": "<think>Thinking</think><answer>42</answer>"}],
            [{"content": self.sample_text_bad}],
        ]
        rewards = soft_format_reward_func(completions, weighting=0.5)

        # The pattern in soft_format_reward_func is r"<think>.*?</think>\s*<answer>.*?</answer>"
        # and uses re.match which requires the pattern to match from the beginning

        # todo: Our sample_text_good starts with "<think>\n" which doesn't match exactly "<think>", this behavior is probably wrong and should be corrected
        self.assertEqual(rewards[0], 0.0)

        # The second example matches the pattern exactly from the beginning
        self.assertEqual(rewards[1], 0.5)

        # Third doesn't match
        self.assertEqual(rewards[2], 0.0)

    def test_xmlcount_reward_func(self):
        """Test the xmlcount_reward_func function, just wraps count_xml"""
        completions = [
            [{"content": self.sample_text_good}],
            [{"content": self.sample_text_bad}],
            [{"content": self.sample_text_partial}],
        ]
        rewards = xmlcount_reward_func(completions, weighting=1.0)
        # First has all XML tags
        self.assertGreater(rewards[0], 0.4)
        # Second has no XML tags
        self.assertEqual(rewards[1], 0.0)
        # Third has some XML tags
        self.assertGreater(rewards[2], 0.0)
        self.assertLess(rewards[2], rewards[0])

    def test_top_k_cumulative_reward(self):
        """Test the top_k_cumulative_reward function"""
        # Patch individual reward functions to return known values
        with (
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.correctness_reward_func"
            ) as mock_correctness,
            patch("hivemind_exp.gsm8k.stage1_rewards.int_reward_func") as mock_int,
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.strict_format_reward_func"
            ) as mock_strict,
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.soft_format_reward_func"
            ) as mock_soft,
            patch("hivemind_exp.gsm8k.stage1_rewards.xmlcount_reward_func") as mock_xml,
        ):
            # Set return values for mocked functions
            mock_correctness.return_value = [2.0, 0.0, 2.0]
            mock_int.return_value = [0.5, 0.0, 0.5]
            mock_strict.return_value = [0.5, 0.0, 0.0]
            mock_soft.return_value = [0.5, 0.0, 0.5]
            mock_xml.return_value = [0.5, 0.0, 0.3]

            # Calculate cumulative rewards
            rewards = top_k_cumulative_reward(
                self.mock_prompts, self.mock_completions, self.mock_answer
            )

            # Check that rewards are summed correctly
            self.assertEqual(rewards[0], 4.0)  # 2.0 + 0.5 + 0.5 + 0.5 + 0.5
            self.assertEqual(rewards[1], 0.0)  # 0.0 + 0.0 + 0.0 + 0.0 + 0.0
            self.assertEqual(rewards[2], 3.3)  # 2.0 + 0.5 + 0.0 + 0.5 + 0.3

    def test_hivemind_cumulative_reward(self):
        """Test the hivemind_cumulative_reward function"""
        # Create mock node
        mock_node = MagicMock(spec=HivemindNode)
        mock_node.key = "test_node"

        # Patch individual reward functions to return known values
        with (
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.correctness_reward_func"
            ) as mock_correctness,
            patch("hivemind_exp.gsm8k.stage1_rewards.int_reward_func") as mock_int,
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.strict_format_reward_func"
            ) as mock_strict,
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.soft_format_reward_func"
            ) as mock_soft,
            patch("hivemind_exp.gsm8k.stage1_rewards.xmlcount_reward_func") as mock_xml,
        ):
            # Set return values for mocked functions
            mock_correctness.return_value = [2.0, 0.0, 2.0]
            mock_int.return_value = [0.5, 0.0, 0.5]
            mock_strict.return_value = [0.5, 0.0, 0.0]
            mock_soft.return_value = [0.5, 0.0, 0.5]
            mock_xml.return_value = [0.5, 0.0, 0.3]

            # Test with max selector
            rewards = hivemind_cumulative_reward(
                mock_node,
                self.mock_prompts,
                self.mock_completions,
                self.mock_answer,
                output_signal_selector="max",
            )

            # Check that the function returns zeros (as expected)
            self.assertEqual(rewards, [0.0, 0.0, 0.0])

            # Check that node.outputs and node.rewards are set correctly
            self.assertEqual(mock_node.rewards, [4.0, 0.0, 3.3])
            self.assertIsNotNone(mock_node.outputs)
            self.assertEqual(
                mock_node.outputs["agent_answers"][mock_node.key], self.sample_text_good
            )

            # Test with None selector
            # The implementation only sets node.outputs and node.rewards when output_signal_selector is not None
            # Looking at the code: if output_signal_selector != None: node.outputs = output_data
            mock_node = MagicMock(spec=HivemindNode)
            mock_node.key = "test_node"

            # Define output_data for None case to avoid UnboundLocalError
            # This is necessary because the actual implementation has a conditional that only defines output_data
            # when output_signal_selector == "max", but then tries to use it if output_signal_selector != None
            with (
                patch(
                    "hivemind_exp.gsm8k.stage1_rewards.correctness_reward_func"
                ) as mock_correctness,
                patch("hivemind_exp.gsm8k.stage1_rewards.int_reward_func") as mock_int,
                patch(
                    "hivemind_exp.gsm8k.stage1_rewards.strict_format_reward_func"
                ) as mock_strict,
                patch(
                    "hivemind_exp.gsm8k.stage1_rewards.soft_format_reward_func"
                ) as mock_soft,
                patch(
                    "hivemind_exp.gsm8k.stage1_rewards.xmlcount_reward_func"
                ) as mock_xml,
            ):
                # Set return values for mocked functions
                mock_correctness.return_value = [2.0, 0.0, 2.0]
                mock_int.return_value = [0.5, 0.0, 0.5]
                mock_strict.return_value = [0.5, 0.0, 0.0]
                mock_soft.return_value = [0.5, 0.0, 0.5]
                mock_xml.return_value = [0.5, 0.0, 0.3]

                rewards = hivemind_cumulative_reward(
                    mock_node,
                    self.mock_prompts,
                    self.mock_completions,
                    self.mock_answer,
                    output_signal_selector=None,
                )

                # Check that node.outputs and node.rewards are not set when output_signal_selector is None
                self.assertFalse(hasattr(mock_node, "outputs"))
                self.assertFalse(hasattr(mock_node, "rewards"))


if __name__ == "__main__":
    unittest.main()
