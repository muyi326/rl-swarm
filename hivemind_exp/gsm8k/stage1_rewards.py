import os
import random
import re

import numpy as np

from hivemind_exp.hivemind_utils import HivemindNode


def extract_xml_answer(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def count_xml(text) -> float:
    if text is None:
        return 0.0
    if not isinstance(text, str):
        return 0.0
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# Reward functions
def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    # Validate inputs
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]
    if answer is None or not answer or not isinstance(answer, list):
        return [0.0] * len(completions)

    try:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
    except (IndexError, KeyError, TypeError):
        # Return default rewards if we can't extract the necessary data
        return [0.0] * len(completions)
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Question:\n{q}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    return [
        1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def int_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 * weighting if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # Validate inputs
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"

    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        # Return default rewards if we can't extract the necessary data
        return [0.0] * len(completions)
    return [1.0 * weighting if match else 0.0 for match in matches]


def soft_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # Validate inputs
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    try:
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
    except (IndexError, KeyError, TypeError):
        # Return default rewards if we can't extract the necessary data
        return [0.0] * len(completions)
    return [1.0 * weighting if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, weighting=1.0, **kwargs) -> list[float]:
    # Validate inputs
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]

    try:
        contents = [completion[0]["content"] for completion in completions]
    except (IndexError, KeyError, TypeError):
        # Return default rewards if we can't extract the necessary data
        return [0.0] * len(completions)
    return [count_xml(c) * weighting for c in contents]


def top_k_cumulative_reward(
    prompts,
    completions,
    answer,
    logging=False,
    **kwargs,
) -> list[float]:
    """
    Dummy reward function that accumulates all rewards into one for prompt generation's top_k selector
    """
    # Validate inputs
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [0.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]

    # Calculate individual rewards
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    int_reward = int_reward_func(completions)
    strict_format_reward = strict_format_reward_func(completions)
    soft_format_reward = soft_format_reward_func(completions)
    xmlcount_reward = xmlcount_reward_func(completions)
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    return total_reward


def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    """
    Dummy reward function that accumulates all rewards into one + saves JSON to node.outputs
    """
    # Validate inputs
    if node is None:
        return [0.0]
    if prompts is None or not prompts or not isinstance(prompts, list):
        return [0.0]
    if completions is None or not completions or not isinstance(completions, list):
        return [0.0]

    # Calculate individual rewards
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    int_reward = int_reward_func(completions)
    strict_format_reward = strict_format_reward_func(completions)
    soft_format_reward = soft_format_reward_func(completions)
    xmlcount_reward = xmlcount_reward_func(completions)
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    if output_signal_selector == "max":
        # Generate output line
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": prompts[0][-1]["content"],
            "answer": answer[0],
            "agent_answers": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
