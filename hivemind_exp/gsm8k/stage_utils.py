import logging
import time
from collections import defaultdict

from hivemind_exp.dht_utils import (
    DHT,
    HivemindNode,
    get_dht_value,
    get_outputs,
    rewards_key,
)
from hivemind_exp.gsm8k.stage_merger import (
    Any,
)
from hivemind_exp.name_utils import get_name_from_peer_id


def merged_prev_stage_datasets(
    dht: DHT,
    node: HivemindNode,
    r: int,
    s: int,
    merge_fn,
    samples_fn,
    dht_sample_limit=200,
    check_interval: float = 5,
    wait_timeout: float = 10,
    log_tag=None,
):
    if not log_tag:
        log_tag = get_name_from_peer_id(node.key)

    logger = logging.getLogger(f"{__name__}:{log_tag}")

    merged_qs = []

    # Retrieves and merges last stage samples locally and from DHT.
    def get_prev_rewards():
        return get_dht_value(dht, key=rewards_key(r, s - 1), beam_size=100)

    prev_rewards: dict[str, Any] | None = get_prev_rewards()
    start_time = time.monotonic()
    while not prev_rewards and time.monotonic() - start_time < wait_timeout:
        logger.info(
            f"Can't retrieve round {r} stage {s - 1} rewards; trying again in {check_interval}s "
        )
        time.sleep(check_interval)
        prev_rewards = get_prev_rewards()

    # Add the current node's local samples first.
    prev_items: dict[str, list] = defaultdict(list)
    try:
        prev_node_outputs = get_outputs(dht, node.key, r, s - 1, node.get_stage_outputs)
        for item in prev_node_outputs.items():
            prev_items[node.key].append(item)
    except ValueError:
        # Joined after the round has started.
        logger.info(f"Could not retrieve local outputs for round {r} stage {s - 1}")

    # Add other nodes' samples iff rewards are available.
    if prev_rewards:
        node_keys = prev_rewards.keys()
        dht_sample_count = 0
        for node_key in node_keys:
            if dht_sample_count > dht_sample_limit:
                break

            if node_key == node.key:
                continue
            try:
                prev_node_outputs = get_outputs(dht, node_key, r, s - 1)
                for item in prev_node_outputs.items():
                    prev_items[node_key].append(item)

                    dht_sample_count += 1
                    if dht_sample_count > dht_sample_limit:
                        break

            except ValueError:
                # Skip this node's answers for the current round and stage.
                logger.debug(
                    f"Found rewards published for node: {node_key} but no outputs!"
                )

    # Group samples by question hash.
    q_to_keyed_items: dict[str, dict[str, Any]] = defaultdict(dict)
    for node_key, items in prev_items.items():
        for item in items:
            q_hash, (_, outputs) = item
            q_to_keyed_items[q_hash][node_key] = outputs

    # Merge sample lists.
    for outputs in q_to_keyed_items.values():
        merged = merge_fn(outputs)
        merged_qs.append(merged)

    return samples_fn(merged_qs)
