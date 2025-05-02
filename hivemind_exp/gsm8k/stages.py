import logging
from collections import defaultdict
from typing import Sequence


import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards
import hivemind_exp.gsm8k.stage3_rewards as stage3_rewards
from hivemind_exp.dht_utils import (
    DHT,
    HivemindNode,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage2_samples, get_stage3_samples
from hivemind_exp.gsm8k.stage_merger import (
    merge_stage1_question,
    merge_stage2_question,
)
from hivemind_exp.gsm8k.stage_utils import merged_prev_stage_datasets
from hivemind_exp.hivemind_utils import SingleStageData, StageData

def gsm8k_stage_data(
    dht: DHT,
    node: HivemindNode,
    initial_train_dataset,
    initial_test_dataset,
    check_interval: float = 5,
    log_tag=None,
):
    def cumulative_reward_0(**kwargs):
        return stage1_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_1(**kwargs):
        return stage2_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_2(**kwargs):
        return stage3_rewards.hivemind_cumulative_reward(node, **kwargs)

    def stage2_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage1_question,
            get_stage2_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    def stage3_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage2_question,
            get_stage3_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    def round_winners(limit=10) -> Sequence[str]:
        final_stage_outputs, _ = merged_prev_stage_datasets(
            dht,
            node,
            node.round_num,
            3,
            lambda x: x,
            lambda v: (v, v),
            check_interval=check_interval,
            log_tag=log_tag,
        )
        logger = logging.getLogger(f"{__name__}:{log_tag}")

        rewards = defaultdict(float)
        for outputs in final_stage_outputs:
            for node_key, output in outputs.items():
                question = output.get("question")
                faulty_completion = False
                if question is None:
                    logger.warning(f"Missing 'question' key in output: {output}")
                    output["question"] = "<no question available>"
                    faulty_completion = True
                stage3_prompt = output.get("stage3_prompt")
                if stage3_prompt is None:
                    logger.warning(f"Missing 'stage3_prompt' key in output: {output}")
                    output["stage3_prompt"] = "<no stage3 prompt available>"
                    faulty_completion = True
                final_answer = output.get("final_agent_decision")
                if final_answer is None:
                    logger.warning(
                        f"Missing 'final_agent_decision' key in output: {output}"
                    )
                    output["final_agent_decision"] = (
                        "<no final agent decision available>"
                    )
                    faulty_completion = True
                prompts = [
                    [
                        {"role": "system", "content": output["question"]},
                        {"role": "system", "content": output["stage3_prompt"]},
                    ],
                ]
                if faulty_completion:
                    rewards[node_key] += 0.0
                else:
                    final_answer = next(iter(output["final_agent_decision"].items()))[1]
                    completions = [[{"role": "assistant", "content": final_answer}]]
                    cumulative_reward_2(
                        prompts=prompts, completions=completions, **output
                    )
                    rewards[node_key] += sum(node.rewards)

        rewards = sorted(list(rewards.items()), key=lambda x: x[1], reverse=True)
        return [n for n, _ in rewards][:limit]

    return StageData(
        round_winner_fn=round_winners,
        stages=[
            SingleStageData(
                name="0",
                reward_funcs=[
                    stage1_rewards.xmlcount_reward_func,
                    stage1_rewards.soft_format_reward_func,
                    stage1_rewards.strict_format_reward_func,
                    stage1_rewards.int_reward_func,
                    stage1_rewards.correctness_reward_func,
                    cumulative_reward_0,
                ],
                datasets_fn=lambda r, s: (initial_train_dataset, initial_test_dataset),  # type: ignore
            ),
            SingleStageData(
                name="1",
                reward_funcs=[
                    stage2_rewards.proper_id_reward_func,
                    stage2_rewards.correctness_reward_func,
                    stage2_rewards.strict_format_reward_func,
                    stage2_rewards.soft_format_reward_func,
                    stage2_rewards.xmlcount_reward_func,
                    cumulative_reward_1,
                ],
                datasets_fn=stage2_datasets_fn,  # type: ignore
            ),
            SingleStageData(
                name="2",
                reward_funcs=[
                    stage3_rewards.consensus_reward_func,
                    stage3_rewards.concensus_correctness_reward_func,
                    stage3_rewards.question_recreation_reward_func,
                    stage3_rewards.final_correctness_reward_func,
                    stage3_rewards.strict_format_reward_func,
                    stage3_rewards.soft_format_reward_func,
                    stage3_rewards.xmlcount_reward_func,
                    cumulative_reward_2,
                ],
                datasets_fn=stage3_datasets_fn,  # type: ignore
            ),
        ],
    )
