from datasets import Dataset, load_dataset

# TODO: Refactor common math components.
from hivemind_exp.gsm8k.generate_prompts import (
    STAGE1_SYSTEM_PROMPT,
    generate_system_prompt,
)


def get_dapo_questions(data) -> Dataset:
    sys_prompt = generate_system_prompt(STAGE1_SYSTEM_PROMPT)

    def map_sample(x):
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["solution"]
        }

    return data.map(map_sample)


def get_stage1_samples(num_samples=100):
    # Load dataset from Hugging Face Hub
    dataset_id = "open-r1/DAPO-Math-17k-Processed"
    dataset: Dataset = load_dataset(dataset_id, "en")["train"] # type: ignore
    datasets = (
        dataset.shuffle(seed=42)
        .select(range(num_samples))
        .train_test_split(test_size=0.5)
    )  # type: ignore
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    # convert our dataset to the r1 prompt
    train_dataset = get_dapo_questions(train_dataset)
    test_dataset = get_dapo_questions(test_dataset)
    return train_dataset, test_dataset


# Reusing stage 2 and 3 sample builders from gsm8k.
