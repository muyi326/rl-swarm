from copy import deepcopy

from hivemind_exp.gsm8k.stage_merger import *
from hivemind_exp.tests.fake_data import *


def test_merge_stage1():
    merged = merge_stage1_question(STAGE_1_OUTPUTS)
    assert merged == STAGE_1_MERGED


def test_merge_stage2():
    merged = merge_stage2_question(STAGE_2_OUTPUTS)
    assert merged == STAGE_2_MERGED


def test_merge_stage1_with_malformed_input():
    # Create a copy of the outputs with a malformed entry
    malformed_outputs = deepcopy(STAGE_1_OUTPUTS)

    # Case 1: Missing required field "question"
    malformed_agent_id = "malformed_agent_1"
    malformed_outputs[malformed_agent_id] = {
        "answer": "42",
        "agent_answers": {"agent1": "The meaning of life is 42."},
    }

    # Case 2: Missing required field "agent_answers"
    malformed_agent_id2 = "malformed_agent_2"
    malformed_outputs[malformed_agent_id2] = {
        "question": "What is the meaning of life?",
        "answer": "42",
    }

    # Merge with malformed inputs
    merged = merge_stage1_question(malformed_outputs)

    # Verify the result still contains valid data from good agents
    assert "question" in merged and merged["question"] is not None
    assert "answer" in merged and merged["answer"] is not None
    assert "agent_answers" in merged

    # Verify malformed agents were skipped (their data not in merged result)
    assert merged["agent_answers"][malformed_agent_id] == "No answer received..."
    assert merged["agent_answers"][malformed_agent_id2] == "No answer received..."

    # Verify good agents' data is still present
    for agent_id in STAGE_1_OUTPUTS.keys():
        assert agent_id in merged["agent_answers"]


def test_merge_stage2_with_malformed_input():
    # Create a copy of the outputs with a malformed entry
    malformed_outputs = deepcopy(STAGE_2_OUTPUTS)

    # Case 1: Missing required field "stage2_prompt"
    malformed_agent_id = "malformed_agent_1"
    malformed_outputs[malformed_agent_id] = {
        "question": "What is the meaning of life?",
        "answer": "42",
        "agent_opinion": {"agent1": "I think the answer is 42."},
    }

    # Case 2: Invalid type for "agent_opinion" (should be dict)
    malformed_agent_id2 = "malformed_agent_2"
    malformed_outputs[malformed_agent_id2] = {
        "question": "What is the meaning of life?",
        "answer": "42",
        "stage2_prompt": "Some prompt text",
        "agent_opinion": "This should be a dict, not a string",
    }

    # Merge with malformed inputs
    merged = merge_stage2_question(malformed_outputs)

    # Verify the result still contains valid data
    assert "question" in merged and merged["question"] is not None
    assert "answer" in merged and merged["answer"] is not None
    assert "stage2_prompt" in merged and merged["stage2_prompt"] is not None
    assert "agent_opinion" in merged

    # Verify malformed agents were skipped (their opinions not in merged result)
    assert merged["agent_opinion"][malformed_agent_id] == "No feedback received..."
    assert merged["agent_opinion"][malformed_agent_id2] == "No feedback received..."

    # Verify good agents' data is still present
    for agent_id in STAGE_2_OUTPUTS.keys():
        assert agent_id in merged["agent_opinion"]


def test_merge_stage1_with_empty_outputs():
    # Test with empty outputs
    merged = merge_stage1_question({})

    # Verify the result has the expected structure with default values
    assert "question" in merged and merged["question"] is None
    assert "answer" in merged and merged["answer"] is None
    assert "agent_answers" in merged and merged["agent_answers"] == {}


def test_merge_stage2_with_empty_outputs():
    # Test with empty outputs
    merged = merge_stage2_question({})

    # Verify the result has the expected structure with default values
    assert "question" in merged and merged["question"] is None
    assert "answer" in merged and merged["answer"] is None
    assert "stage2_prompt" in merged and merged["stage2_prompt"] is None
    assert "agent_opinion" in merged and merged["agent_opinion"] == {}
