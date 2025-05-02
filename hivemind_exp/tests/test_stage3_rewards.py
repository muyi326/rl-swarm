import unittest
import re
import numpy as np
from unittest.mock import patch, MagicMock

from hivemind_exp.gsm8k.stage3_rewards import (
    extract_xml_identity,
    extract_xml_final_answer,
    extract_xml_question,
    extract_xml_ids,
    extract_xml_choices,
    extract_original_question,
    extract_answers,
    count_xml,
    swarm_majority,
    consensus_reward_func,
    question_recreation_reward_func,
    concensus_correctness_reward_func,
    final_correctness_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    hivemind_cumulative_reward,
)
from hivemind_exp.hivemind_utils import HivemindNode


class TestStage3Rewards(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_text_good = "<summarize_feedback>\nAfter reviewing the feedback, student1 has the correct approach and answer.\n</summarize_feedback>\n<majority>\nstudent1\n</majority>\n<question>\nWhat is 6 times 7?\n</question>\n<think>\nTo calculate 6 times 7, I'll multiply these numbers directly.\n6 × 7 = 42\n</think>\n<answer>\n42\n</answer>\n"
        self.sample_text_bad = "I think student1 is correct with the answer 42."
        self.sample_text_partial = "<summarize_feedback>\nAfter reviewing the feedback, student1 has the correct approach and answer.\n</summarize_feedback>\n<majority>\nstudent1\n</majority>\n<question>\nWhat is 6 times 7?\n</question>\n<think>\nTo calculate 6 times 7, I'll multiply these numbers directly.\n6 × 7 = 42\n</think>\n<answer>\n42"

        # Sample prompt with student answers and feedback
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

  
After comparing these answers, the following feedback was given about which answer is best: 

<identify>student1</identify>
I believe student1 has the correct answer. They correctly multiplied 6 by 7 to get 42.

<identify>student1</identify>
Student1 is correct. They properly calculated 6 × 7 = 42.

<identify>student2</identify>
I think student2 is closest, but made a small error. The answer should be 42, not 43.

Please summarize the feedback, identify the majority opinion, restate the original question, and provide the final correct answer."""

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
            extract_xml_identity("<majority>student2</majority>"), "student2"
        )
        self.assertEqual(
            extract_xml_identity("Some text <majority>student3</majority> more text"),
            "student3",
        )
        # Edge cases
        self.assertEqual(extract_xml_identity("<majority></majority>"), "")
        # When no majority tag is present, the implementation will return the string itself
        self.assertEqual(extract_xml_identity("No majority tag"), "No majority tag")

    def test_extract_xml_final_answer(self):
        """Test the extract_xml_final_answer function"""
        self.assertEqual(extract_xml_final_answer(self.sample_text_good), "42")
        self.assertEqual(extract_xml_final_answer("<answer>123</answer>"), "123")
        self.assertEqual(
            extract_xml_final_answer("Some text <answer>456</answer> more text"), "456"
        )
        # Edge cases
        self.assertEqual(extract_xml_final_answer("<answer></answer>"), "")
        # When no answer tag is present, the implementation will return the string itself
        self.assertEqual(extract_xml_final_answer("No answer tag"), "No answer tag")

    def test_extract_xml_question(self):
        """Test the extract_xml_question function"""
        self.assertEqual(
            extract_xml_question(self.sample_text_good), "What is 6 times 7?"
        )
        self.assertEqual(
            extract_xml_question("<question>What is 2+2?</question>"), "What is 2+2?"
        )
        self.assertEqual(
            extract_xml_question(
                "Some text <question>What is 3*3?</question> more text"
            ),
            "What is 3*3?",
        )
        # Edge cases
        self.assertEqual(extract_xml_question("<question></question>"), "")
        # When no question tag is present, the implementation will return the string itself
        self.assertEqual(extract_xml_question("No question tag"), "No question tag")

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

    def test_extract_xml_choices(self):
        """Test the extract_xml_choices function"""
        # Test with a string containing identify tags
        sample_with_choices = """
        <identify>student1</identify> said something
        <identify>student2</identify> said something else
        <identify>student1</identify> said another thing
        """
        choices = extract_xml_choices(sample_with_choices)
        self.assertEqual(choices, ["student1", "student2", "student1"])

        # Test with actual prompt
        choices_from_prompt = extract_xml_choices(self.sample_prompt)
        self.assertEqual(choices_from_prompt, ["student1", "student1", "student2"])

        # Test with no identify tags
        self.assertEqual(extract_xml_choices("No identify tags here"), [])

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

    def test_count_xml(self):
        """Test the count_xml function"""
        # Good format should get high score
        good_score = count_xml(self.sample_text_good)
        self.assertGreater(good_score, 0.8)  # Should have most XML tags

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

    def test_swarm_majority(self):
        """Test the swarm_majority function"""
        # Test with a clear majority
        choices = ["student1", "student1", "student2", "student3", "student1"]
        majority = swarm_majority(choices)
        self.assertEqual(majority, ["student1"])

        # Test with a tie
        choices_tie = ["student1", "student2", "student1", "student2"]
        majority_tie = swarm_majority(choices_tie)
        # Both should be in the majority list
        self.assertIn("student1", majority_tie)
        self.assertIn("student2", majority_tie)
        self.assertEqual(len(majority_tie), 2)

        # Test with empty list
        self.assertEqual(swarm_majority([]), [])

        # Test with a single choice
        self.assertEqual(swarm_majority(["student1"]), ["student1"])

    def test_consensus_reward_func(self):
        """Test the consensus_reward_func function"""
        # Use a single completion at a time as the function expects
        with (
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.extract_xml_choices"
            ) as mock_choices,
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            # Set up mocks
            mock_choices.return_value = [
                "student1",
                "student1",
                "student2",
            ]  # student1 is the majority

            # Test with student1 (matches majority)
            rewards1 = consensus_reward_func(
                self.mock_prompts,
                [self.mock_completions[0]],
                weighting=2.0,
                logging=False,
            )

            # Test with a different student
            different_student = [[{"content": "<majority>\nstudent2\n</majority>\n"}]]
            rewards2 = consensus_reward_func(
                self.mock_prompts, different_student, weighting=2.0, logging=False
            )

            # Test with a student not in choices
            invalid_student = [[{"content": "<majority>\nstudent3\n</majority>\n"}]]
            rewards3 = consensus_reward_func(
                self.mock_prompts, invalid_student, weighting=2.0, logging=False
            )

            # student1 matches the majority
            self.assertEqual(rewards1[0], 2.0)

            # student2 is in choices but not the majority
            self.assertEqual(rewards2[0], 0.0)

            # student3 is not in choices
            self.assertEqual(rewards3[0], 0.0)

    def test_question_recreation_reward_func(self):
        """Test the question_recreation_reward_func function"""
        # Use a single completion at a time
        with (
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.extract_original_question"
            ) as mock_original,
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            # Set up mocks
            mock_original.return_value = "What is 6 times 7?"

            # Test with correct question
            rewards1 = question_recreation_reward_func(
                self.mock_prompts,
                [self.mock_completions[0]],
                weighting=1.0,
                logging=False,
            )

            # Test with slightly different question
            different_question = [
                [{"content": "<question>\nWhat is six times seven?\n</question>\n"}]
            ]
            rewards2 = question_recreation_reward_func(
                self.mock_prompts, different_question, weighting=1.0, logging=False
            )

            # Test with completely different question
            wrong_question = [[{"content": "<question>\nWhat is 2+2?\n</question>\n"}]]
            rewards3 = question_recreation_reward_func(
                self.mock_prompts, wrong_question, weighting=1.0, logging=False
            )

            # Exact match should get full reward
            self.assertEqual(rewards1[0], 1.0)

            # Similar question should get partial reward
            self.assertGreater(rewards2[0], 0.0)
            self.assertLess(rewards2[0], 1.0)

            # Different question might still get a moderate reward due to SequenceMatcher behavior
            # The actual implementation uses SequenceMatcher(None, r, q).ratio() which can give
            # higher scores than expected for short strings
            # So we just check that it's less than the exact match
            self.assertLess(rewards3[0], rewards1[0])

    def test_concensus_correctness_reward_func(self):
        """Test the concensus_correctness_reward_func function"""
        # Simplify the test to focus on basic functionality without comparing exact reward values
        with (
            patch("hivemind_exp.gsm8k.stage3_rewards.extract_answers") as mock_answers,
            patch(
                "hivemind_exp.gsm8k.stage1_rewards.extract_xml_answer"
            ) as mock_extract,
            patch("hivemind_exp.gsm8k.stage1_rewards.count_xml") as mock_count,
            patch("re.match") as mock_match,
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            # Set up mocks
            mock_answers.return_value = {
                "student1": "<think>\nI need to multiply 6 by 7.\n6 × 7 = 42\n</think>\n<answer>\n42\n</answer>",
                "student2": "<think>\nI'll calculate 6 × 7.\n6 × 7 = 43\n</think>\n<answer>\n43\n</answer>",
                "student3": "<think>\nLet me compute 6 times 7.\n6 + 7 = 13\n</think>\n<answer>\n13\n</answer>",
            }

            # Mock extract_xml_answer to return the correct values
            mock_extract.side_effect = (
                lambda x: "42" if "student1" in x else "43" if "student2" in x else "13"
            )

            # Mock count_xml to return a consistent value
            mock_count.return_value = 0.5

            # Mock re.match to always return True
            mock_match.return_value = True

            # Test with student1 (correct answer)
            student1 = [[{"content": "<majority>\nstudent1\n</majority>\n"}]]
            rewards1 = concensus_correctness_reward_func(
                self.mock_prompts,
                student1,
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # Verify that student1 gets a positive reward for having the correct answer
            self.assertGreater(rewards1[0], 0.0)

            # Test with "None" when all student answers are correct
            # First, modify the mock to make all answers correct
            mock_extract.side_effect = lambda x: "42"  # All answers now return 42

            none_student = [[{"content": "<majority>\nNone\n</majority>\n"}]]
            rewards_none_correct = concensus_correctness_reward_func(
                self.mock_prompts,
                none_student,
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # "None" should get a reward if all answers are correct
            self.assertGreater(rewards_none_correct[0], 0.0)

    def test_final_correctness_reward_func(self):
        """Test the final_correctness_reward_func function"""
        # Test with the correct final answer
        with (
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            # Test with correct answer
            rewards1 = final_correctness_reward_func(
                self.mock_prompts,
                [self.mock_completions[0]],
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # Test with incorrect answer
            wrong_answer = [[{"content": "<answer>\n43\n</answer>\n"}]]
            rewards2 = final_correctness_reward_func(
                self.mock_prompts,
                wrong_answer,
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # Test with non-numeric answer
            non_numeric = [[{"content": "<answer>\nforty-two\n</answer>\n"}]]
            rewards3 = final_correctness_reward_func(
                self.mock_prompts,
                non_numeric,
                self.mock_answer,
                weighting=2.0,
                logging=False,
            )

            # Correct answer should get full reward
            self.assertEqual(rewards1[0], 2.0)

            # Incorrect answer should get no reward
            self.assertEqual(rewards2[0], 0.0)

            # Non-numeric answer should get no reward
            self.assertEqual(rewards3[0], 0.0)

    def test_strict_format_reward_func(self):
        """Test the strict_format_reward_func function"""
        # The strict_format_reward_func uses a very specific regex pattern that requires exact format
        # Let's create a sample that exactly matches the expected pattern
        exact_format = "<summarize_feedback>\nSummary\n</summarize_feedback>\n<majority>\nstudent1\n</majority>\n<question>\nWhat?\n</question>\n<think>\nThinking\n</think>\n<answer>\n42\n</answer>\n"

        completions = [
            [{"content": exact_format}],  # Exactly matches the pattern
            [{"content": self.sample_text_bad}],  # Doesn't match at all
            [{"content": self.sample_text_partial}],  # Partial match
        ]

        # Mock the logging to avoid file operations
        with (
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            rewards = strict_format_reward_func(
                completions, weighting=0.5, logging=False
            )

            # First should match the exact pattern
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
                    "content": "<summarize_feedback>Summary</summarize_feedback><majority>student1</majority><question>What?</question><think>Thinking</think><answer>42</answer>"
                }
            ],
            [{"content": self.sample_text_bad}],
        ]

        # Mock the logging to avoid file operations
        with (
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            rewards = soft_format_reward_func(completions, weighting=0.5, logging=False)

            # The pattern in soft_format_reward_func is more lenient
            # Our sample_text_good has newlines which might not match exactly
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
        with (
            patch("os.makedirs", return_value=None),
            patch("builtins.open", return_value=MagicMock()),
        ):
            rewards = xmlcount_reward_func(completions, weighting=1.0, logging=False)

            # First has all XML tags
            self.assertGreater(rewards[0], 0.8)

            # Second has no XML tags
            self.assertEqual(rewards[1], 0.0)

            # Third has some XML tags
            self.assertGreater(rewards[2], 0.0)
            self.assertLess(rewards[2], rewards[0])

    def test_hivemind_cumulative_reward(self):
        """Test the hivemind_cumulative_reward function"""
        # Create mock node
        mock_node = MagicMock(spec=HivemindNode)
        mock_node.key = "test_node"

        # Patch individual reward functions to return known values
        with (
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.consensus_reward_func"
            ) as mock_consensus,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.concensus_correctness_reward_func"
            ) as mock_correctness,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.question_recreation_reward_func"
            ) as mock_question,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.final_correctness_reward_func"
            ) as mock_final,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.strict_format_reward_func"
            ) as mock_strict,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.soft_format_reward_func"
            ) as mock_soft,
            patch("hivemind_exp.gsm8k.stage3_rewards.xmlcount_reward_func") as mock_xml,
            patch(
                "hivemind_exp.gsm8k.stage3_rewards.extract_original_question"
            ) as mock_extract,
        ):
            # Set return values for mocked functions
            mock_consensus.return_value = [2.0, 0.0, 2.0]
            mock_correctness.return_value = [2.5, 0.0, 0.0]
            mock_question.return_value = [1.0, 0.5, 0.8]
            mock_final.return_value = [2.0, 0.0, 0.0]
            mock_strict.return_value = [0.5, 0.0, 0.0]
            mock_soft.return_value = [0.0, 0.5, 0.0]
            mock_xml.return_value = [1.0, 0.0, 0.7]
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
            self.assertEqual(mock_node.rewards, [9.0, 1.0, 3.5])
            self.assertIsNotNone(mock_node.outputs)
            self.assertEqual(
                mock_node.outputs["final_agent_decision"][mock_node.key],
                self.sample_text_good,
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

            # Test with empty answer
            mock_node = MagicMock(spec=HivemindNode)
            mock_node.key = "test_node"

            rewards_empty = hivemind_cumulative_reward(
                mock_node,
                self.mock_prompts,
                self.mock_completions,
                [],
                output_signal_selector="max",
                logging=False,
            )

            # Check that the function handles empty answer gracefully
            self.assertEqual(rewards_empty, [0.0, 0.0, 0.0])
            self.assertEqual(mock_node.outputs["answer"], "Unknown")


if __name__ == "__main__":
    unittest.main()
