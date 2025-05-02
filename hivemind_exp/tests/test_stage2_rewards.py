import unittest
import re
import numpy as np
from unittest.mock import patch, MagicMock

from hivemind_exp.gsm8k.stage2_rewards import (
    extract_xml_identity,
    extract_xml_ids,
    extract_original_question,
    extract_answers,
    count_xml,
    proper_id_reward_func,
    correctness_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    top_k_cumulative_reward,
    hivemind_cumulative_reward,
)
from hivemind_exp.hivemind_utils import HivemindNode


class TestStage2Rewards(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_text_good = "<compare>\nLet me compare the answers.\n</compare>\n<explain>\nAfter reviewing the answers, I can see that student1 has the correct approach.\n</explain>\n<identify>\nstudent1\n</identify>\n"
        self.sample_text_bad = "I think student1 is correct."
        self.sample_text_partial = "<compare>\nLet me compare the answers.\n</compare>\n<explain>\nAfter reviewing the answers, I can see that student1 has the correct approach.\n</explain>\n<identify>\nstudent1"

        # Sample prompt with student answers
        self.sample_prompt = """The question we were given is: What is 6 times 7?  

The following answers to this question were suggested:

<student>student1</student> said 
<think>
I need to multiply 6 by 7.
6 × 7 = 42
</think>
<answer>
42
</answer>

<student>student2</student> said 
<think>
I'll calculate 6 × 7.
6 × 7 = 43
</think>
<answer>
43
</answer>

<student>student3</student> said 
<think>
Let me compute 6 times 7.
6 + 7 = 13
</think>
<answer>
13
</answer>

Which student's answer is correct? Analyze each answer carefully."""

        # Mock data for reward functions
        self.mock_prompts = [[{"content": self.sample_prompt}]]
        self.mock_completions = [
            [{"content": self.sample_text_good}],
            [{"content": self.sample_text_bad}],
            [{"content": self.sample_text_partial}],
        ]
        self.mock_answer = ["42"]

    def test_extract_xml_identity(self):
        """Test the extract_xml_identity function"""
        self.assertEqual(extract_xml_identity(self.sample_text_good), "student1")
        self.assertEqual(
            extract_xml_identity("<identify>student2</identify>"), "student2"
        )
        self.assertEqual(
            extract_xml_identity("Some text <identify>student3</identify> more text"),
            "student3",
        )
        # Edge cases
        self.assertEqual(extract_xml_identity("<identify></identify>"), "")
        # todo: When no identify tag is present, the implementation will return the string itself
        # as text.split("</identify>")[0] will be the original string, this behavior is probably wrong and should be corrected
        self.assertEqual(extract_xml_identity("No identify tag"), "No identify tag")

    def test_extract_xml_ids(self):
        """Test the extract_xml_ids function"""
        # Test with a string containing student tags
        sample_with_students = """
        <student>student1</student> said something
        <student>student2</student> said something else
        <student>student3</student> said another thing
        """
        ids = extract_xml_ids(sample_with_students)
        self.assertEqual(ids, ["student1", "student2", "student3"])

        # Test with actual prompt
        ids_from_prompt = extract_xml_ids(self.sample_prompt)
        self.assertEqual(ids_from_prompt, ["student1", "student2", "student3"])

        # Test with no student tags
        self.assertEqual(extract_xml_ids("No student tags here"), [])

        # Test with non-string input
        self.assertEqual(extract_xml_ids(None), [])

    def test_extract_original_question(self):
        """Test the extract_original_question function"""
        # Test with the sample prompt
        question = extract_original_question(self.sample_prompt)
        self.assertEqual(question, "What is 6 times 7?")

        # Test with a different format
        different_format = "The question we were given is: What is 2+2?  \n\nThe following answers to this question were suggested:"
        self.assertEqual(extract_original_question(different_format), "What is 2+2?")

        # Test with no question format
        # The implementation will return the input if the expected format isn't found
        no_question = "No question here"
        result = extract_original_question(no_question)
        self.assertEqual(result, no_question)

    def test_extract_answers(self):
        """Test the extract_answers function"""
        # Test with the sample prompt
        answers = extract_answers(self.sample_prompt)
        self.assertEqual(len(answers), 3)
        self.assertIn("student1", answers)
        self.assertIn("student2", answers)
        self.assertIn("student3", answers)

        # Check the content of the answers
        self.assertIn("<think>", answers["student1"])
        self.assertIn("<answer>", answers["student1"])
        self.assertIn("42", answers["student1"])
        self.assertIn("43", answers["student2"])
        self.assertIn("13", answers["student3"])

        # Test with no student tags
        self.assertEqual(extract_answers("No student tags here"), {})

        # Test with non-string input
        self.assertEqual(extract_answers(None), {})

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

        # Test with extra text after </identify>
        text_with_extra = self.sample_text_good + "extra text"
        self.assertLess(count_xml(text_with_extra), good_score)

    def test_proper_id_reward_func(self):
        """Test the proper_id_reward_func function"""
        # Use a single completion at a time as the function expects
        rewards1 = proper_id_reward_func(
            self.mock_prompts,
            [self.mock_completions[0]],
            self.mock_answer,
            weighting=2.0,
            logging=False,
        )
        rewards2 = proper_id_reward_func(
            self.mock_prompts,
            [self.mock_completions[1]],
            self.mock_answer,
            weighting=2.0,
            logging=False,
        )
        rewards3 = proper_id_reward_func(
            self.mock_prompts,
            [self.mock_completions[2]],
            self.mock_answer,
            weighting=2.0,
            logging=False,
        )

        # First completion has a valid student ID
        self.assertEqual(rewards1[0], 2.0)

        # Second completion doesn't have ID in XML format
        self.assertEqual(rewards2[0], 0.0)

        # Third completion has valid ID but incomplete format
        self.assertEqual(rewards3[0], 2.0)

        # Test with an invalid student ID
        invalid_completion = [
            [{"content": "<identify>\ninvalid_student\n</identify>\n"}]
        ]
        rewards_invalid = proper_id_reward_func(
            self.mock_prompts,
            invalid_completion,
            self.mock_answer,
            weighting=2.0,
            logging=False,
        )
        self.assertEqual(rewards_invalid[0], 0.0)

    def test_correctness_reward_func(self):
        """Test the correctness_reward_func function"""
        # The correctness_reward_func is complex and has multiple conditions
        # Let's test it with more controlled mocking

        # Create a controlled test environment
        with (
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.extract_xml_answer"
            ) as mock_extract,
            patch("hivemind_exp.gsm8k.stage1_rewards.count_xml") as mock_count,
            patch("re.match") as mock_match,
        ):
            # Set up mocks
            mock_extract.side_effect = (
                lambda x: "42" if "student1" in x else "43" if "student2" in x else "13"
            )
            mock_count.return_value = 0.5
            mock_match.return_value = True  # Make all regex matches succeed

            # Test with student1 (correct answer)
            rewards1 = correctness_reward_func(
                self.mock_prompts,
                [self.mock_completions[0]],
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # Student1 has correct answer (42) and should get a high reward
            self.assertGreater(rewards1[0], 0.0)

            # Test with "None" when there's a correct answer (student1)
            none_student = [[{"content": "<identify>\nNone\n</identify>\n"}]]
            rewards_none = correctness_reward_func(
                self.mock_prompts,
                none_student,
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # "None" should get a reward only if all answers are correct (not if all are wrong)
            # Looking at the implementation, the "None" option gets a reward when:
            # all(check_submissions) where check_submissions is [True if r == a else False for r, a in zip(agent_as, answer)]
            # In our case, student1 has the correct answer but others don't, so "None" should get 0
            self.assertEqual(rewards_none[0], 0.0)

            # Now test with "None" when all student answers are correct
            with patch(
                "hivemind_exp.gsm8k.stage2_rewards.extract_answers"
            ) as mock_answers:
                # Make all answers correct
                mock_answers.return_value = {
                    "student1": "<answer>\n42\n</answer>",
                    "student2": "<answer>\n42\n</answer>",
                    "student3": "<answer>\n42\n</answer>",
                }

                # We also need to patch the extract_xml_answer to return consistent values
                mock_extract.side_effect = lambda x: "42"  # All answers are now 42

                rewards_none_correct = correctness_reward_func(
                    self.mock_prompts,
                    none_student,
                    self.mock_answer,
                    weighting=2.0,
                    logging=False,
                )

                # Now "None" should get a reward since all answers are correct
                self.assertGreater(rewards_none_correct[0], 0.0)

    def test_strict_format_reward_func(self):
        """Test the strict_format_reward_func function"""
        completions = [
            [{"content": self.sample_text_good}],
            [{"content": self.sample_text_bad}],
            [{"content": self.sample_text_partial}],
        ]

        # Mock the logging to avoid file operations
        with patch("os.makedirs"), patch("builtins.open"):
            rewards = strict_format_reward_func(
                completions, weighting=0.5, logging=False
            )

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
            [
                {
                    "content": "<compare>Comparing</compare><explain>Explaining</explain><identify>student1</identify>"
                }
            ],
            [{"content": self.sample_text_bad}],
        ]

        # Mock the logging to avoid file operations
        with patch("os.makedirs"), patch("builtins.open"):
            rewards = soft_format_reward_func(completions, weighting=0.5, logging=False)

            # The pattern in soft_format_reward_func is more lenient
            # Our sample_text_good has newlines which might not match exactly
            # This behavior is probably wrong and should be corrected
            self.assertEqual(rewards[0], 0.0)

            # The second example matches the pattern exactly
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

        # Mock the logging to avoid file operations
        with patch("os.makedirs"), patch("builtins.open"):
            rewards = xmlcount_reward_func(completions, weighting=1.0, logging=False)

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
                "hivemind_exp.gsm8k.stage2_rewards.proper_id_reward_func"
            ) as mock_proper_id,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.correctness_reward_func"
            ) as mock_correctness,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.strict_format_reward_func"
            ) as mock_strict,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.soft_format_reward_func"
            ) as mock_soft,
            patch("hivemind_exp.gsm8k.stage2_rewards.xmlcount_reward_func") as mock_xml,
        ):
            # Set return values for mocked functions
            mock_proper_id.return_value = [2.0, 0.0, 2.0]
            mock_correctness.return_value = [2.5, 0.0, 0.0]
            mock_strict.return_value = [0.5, 0.0, 0.0]
            mock_soft.return_value = [0.0, 0.5, 0.0]
            mock_xml.return_value = [0.5, 0.0, 0.3]

            # Calculate cumulative rewards
            rewards = top_k_cumulative_reward(
                self.mock_prompts,
                self.mock_completions,
                self.mock_answer,
                logging=False,
            )

            # Check that rewards are summed correctly
            self.assertEqual(rewards[0], 5.5)  # 2.0 + 2.5 + 0.5 + 0.0 + 0.5
            self.assertEqual(rewards[1], 0.5)  # 0.0 + 0.0 + 0.0 + 0.5 + 0.0
            self.assertEqual(rewards[2], 2.3)  # 2.0 + 0.0 + 0.0 + 0.0 + 0.3

    def test_hivemind_cumulative_reward(self):
        """Test the hivemind_cumulative_reward function"""
        # Create mock node
        mock_node = MagicMock(spec=HivemindNode)
        mock_node.key = "test_node"

        # Patch individual reward functions to return known values
        with (
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.proper_id_reward_func"
            ) as mock_proper_id,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.correctness_reward_func"
            ) as mock_correctness,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.strict_format_reward_func"
            ) as mock_strict,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.soft_format_reward_func"
            ) as mock_soft,
            patch("hivemind_exp.gsm8k.stage2_rewards.xmlcount_reward_func") as mock_xml,
            patch(
                "hivemind_exp.gsm8k.stage2_rewards.extract_original_question"
            ) as mock_extract,
        ):
            # Set return values for mocked functions
            mock_proper_id.return_value = [2.0, 0.0, 2.0]
            mock_correctness.return_value = [2.5, 0.0, 0.0]
            mock_strict.return_value = [0.5, 0.0, 0.0]
            mock_soft.return_value = [0.0, 0.5, 0.0]
            mock_xml.return_value = [0.5, 0.0, 0.3]
            mock_extract.return_value = "What is 6 times 7?"

            # Test with max selector
            rewards = hivemind_cumulative_reward(
                mock_node,
                self.mock_prompts,
                self.mock_completions,
                self.mock_answer,
                output_signal_selector="max",
                logging=False,
            )

            # Check that the function returns zeros (as expected)
            self.assertEqual(rewards, [0.0, 0.0, 0.0])

            # Check that node.outputs and node.rewards are set correctly
            self.assertEqual(mock_node.rewards, [5.5, 0.5, 2.3])
            self.assertIsNotNone(mock_node.outputs)
            self.assertEqual(
                mock_node.outputs["agent_opinion"][mock_node.key], self.sample_text_good
            )
            self.assertEqual(mock_node.outputs["question"], "What is 6 times 7?")
            self.assertEqual(mock_node.outputs["answer"], "42")

            # Test with None selector
            # The implementation only sets node.outputs and node.rewards when output_signal_selector is not None
            mock_node = MagicMock(spec=HivemindNode)
            mock_node.key = "test_node"

            rewards = hivemind_cumulative_reward(
                mock_node,
                self.mock_prompts,
                self.mock_completions,
                self.mock_answer,
                output_signal_selector=None,
                logging=False,
            )

            # Check that node.outputs and node.rewards are not set when output_signal_selector is None
            self.assertFalse(hasattr(mock_node, "outputs"))
            self.assertFalse(hasattr(mock_node, "rewards"))


if __name__ == "__main__":
    unittest.main()
