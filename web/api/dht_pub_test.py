import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from pythonjsonlogger import jsonlogger

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Note: We're patching hivemind_exp functions (rewards_key, get_dht_value, get_name_from_peer_id)
# because these functions are only copied over at build time in Docker and aren't available during local testing.
# This allows us to test the DHTPublisher class without needing the actual hivemind_exp module.

from web.api.dht_pub import GossipDHTPublisher, RewardsDHTPublisher
from web.api.kinesis import RewardsMessage, RewardsMessageData


class TestRewardsDHTPublisher:
    """Tests for the RewardsDHTPublisher class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_dht = MagicMock()
        self.mock_kinesis = MagicMock()

        # Create a real logger for testing
        self.mock_logger = logging.getLogger("test_logger")
        self.mock_logger.setLevel(logging.INFO)

        # Add a handler to the logger so caplog can capture the logs
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Use the JSON formatter from python-json-logger
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(message)s %(extra)s"
        )
        handler.setFormatter(json_formatter)
        self.mock_logger.addHandler(handler)

        self.coordinator = MagicMock()

        # Create the publisher with a short poll interval for testing
        self.publisher = RewardsDHTPublisher(
            dht=self.mock_dht,
            kinesis_client=self.mock_kinesis,
            logger=self.mock_logger,
            poll_interval_seconds=0.1,
            coordinator=self.coordinator,
        )

    def teardown_method(self):
        """Clean up after tests."""
        # Stop the publisher if it's running
        if self.publisher.running:
            self.publisher.stop()
            # Give it a moment to stop
            time.sleep(0.2)

    def test_initialization(self, caplog):
        """Test that the publisher initializes correctly."""
        # Set the caplog level to capture INFO messages
        caplog.set_level(logging.INFO)

        # Re-initialize the publisher to capture the logs
        self.publisher = RewardsDHTPublisher(
            dht=self.mock_dht,
            kinesis_client=self.mock_kinesis,
            logger=self.mock_logger,
            poll_interval_seconds=0.1,
            coordinator=self.coordinator,
        )

        # Verify the log message was captured
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "RewardsDHTPublisher initialized"

        # Verify other properties
        assert self.publisher.dht == self.mock_dht
        assert self.publisher.kinesis_client == self.mock_kinesis
        assert self.publisher.coordinator == self.coordinator
        assert self.publisher.poll_interval_seconds == 0.1
        assert self.publisher.current_round == -1
        assert self.publisher.current_stage == -1
        assert self.publisher.last_polled == None
        assert self.publisher.running == False
        assert self.publisher._poll_thread == None

    def test_poll_once_no_change(self, caplog):
        """Test polling when there's no round/stage change."""
        # Set up the coordinator mock to return the same round/stage
        self.coordinator.get_round_and_stage.return_value = (1, 1)
        self.publisher.current_round = 1
        self.publisher.current_stage = 1

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that the round/stage didn't change
        assert self.publisher.current_round == 1
        assert self.publisher.current_stage == 1

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Polled for round/stage"
        assert caplog.records[0].round == 1
        assert caplog.records[0].stage == 1

        # Check that last_polled was updated
        assert self.publisher.last_polled is not None

    def test_poll_once_with_change(self, caplog):
        """Test polling when there's a round/stage change."""
        # Set up the coordinator mock to return a different round/stage
        self.coordinator.get_round_and_stage.return_value = (2, 1)
        self.publisher.current_round = 1
        self.publisher.current_stage = 1

        # Mock the _publish_rewards method
        self.publisher._publish_rewards = MagicMock()

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that the round/stage changed
        assert self.publisher.current_round == 2
        assert self.publisher.current_stage == 1

        # Check that _publish_rewards was called for old round/stage
        self.publisher._publish_rewards.assert_has_calls(
            [
                call(1, 1),  # Old round/stage
            ]
        )

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Polled for round/stage"
        assert caplog.records[0].round == 2
        assert caplog.records[0].stage == 1

        assert caplog.records[1].message == "Round or stage changed"
        assert caplog.records[1].old_round == 1
        assert caplog.records[1].old_stage == 1
        assert caplog.records[1].new_round == 2
        assert caplog.records[1].new_stage == 1

        # Check that last_polled was updated
        assert self.publisher.last_polled is not None

    def test_poll_once_error(self, caplog):
        """Test polling when there's an error."""
        # Set up the coordinator mock to raise an exception
        self.coordinator.get_round_and_stage.side_effect = Exception("Test error")

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that the round/stage didn't change
        assert self.publisher.current_round == -1
        assert self.publisher.current_stage == -1

        # Check that the logger was called with the error
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Error polling for round/stage in rewards"
        assert caplog.records[0].error == "Test error"

        # Check that last_polled was not updated
        assert self.publisher.last_polled is None

    def test_publish_rewards(self, caplog):
        """Test publishing rewards."""
        # Set up test data
        round_num = 1
        stage_num = 1
        rewards_data = {"peer_id_1": 0.5, "peer_id_2": 0.3}

        # Mock the _get_rewards_data method to return our test data
        self.publisher._get_rewards_data = MagicMock(return_value=rewards_data)

        # Mock the _get_peer_name_from_id method
        self.publisher._get_peer_name_from_id = MagicMock(
            side_effect=["name1", "name2"]
        )

        # Mock the Kinesis client's put_rewards method
        self.publisher.kinesis_client.put_rewards = MagicMock()

        # Publish rewards
        self.publisher._publish_rewards(round_num, stage_num)

        # Check that _get_rewards_data was called with the correct arguments
        self.publisher._get_rewards_data.assert_called_once_with(round_num, stage_num)

        # Check that _get_peer_name_from_id was called for each peer ID
        self.publisher._get_peer_name_from_id.assert_any_call("peer_id_1")
        self.publisher._get_peer_name_from_id.assert_any_call("peer_id_2")

        # Check that put_rewards was called
        self.publisher.kinesis_client.put_rewards.assert_called_once()

        # Get the actual message that was passed to put_rewards
        actual_message = self.publisher.kinesis_client.put_rewards.call_args[0][0]

        # Check that the message has the correct type
        assert actual_message.type == "rewards"

        # Check that the message has the correct data
        assert len(actual_message.data) == 2

        # Check the first data item
        assert actual_message.data[0].peer_id == "peer_id_1"
        assert actual_message.data[0].peer_name == "name1"
        assert actual_message.data[0].amount == 0.5
        assert actual_message.data[0].round == round_num
        assert actual_message.data[0].stage == stage_num

        # Check the second data item
        assert actual_message.data[1].peer_id == "peer_id_2"
        assert actual_message.data[1].peer_name == "name2"
        assert actual_message.data[1].amount == 0.3
        assert actual_message.data[1].round == round_num
        assert actual_message.data[1].stage == stage_num

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Publishing rewards"
        assert caplog.records[0].round == round_num
        assert caplog.records[0].stage == stage_num
        assert caplog.records[0].num_peers == 2

    def test_publish_rewards_no_data(self, caplog):
        """Test publishing rewards when there's no data."""
        # Set up test data
        round_num = 1
        stage_num = 1

        self.publisher._get_rewards_data = MagicMock(return_value=None)

        # Mock the _create_rewards_message method
        self.publisher._create_rewards_message = MagicMock()

        # Publish rewards
        self.publisher._publish_rewards(round_num, stage_num)

        # Check that _create_rewards_message was not called
        assert self.publisher._create_rewards_message.call_count == 0

        # Check that put_rewards was not called
        assert self.mock_kinesis.put_rewards.call_count == 0

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "No rewards data for round, stage"
        assert caplog.records[0].round == round_num
        assert caplog.records[0].stage == stage_num

    @patch("web.api.dht_pub.get_name_from_peer_id")
    def test_create_rewards_message(self, mock_get_name, caplog):
        """Test creating a rewards message."""
        # Set up test data
        round_num = 1
        stage_num = 1
        rewards_data = {"peer_id_1": 0.5, "peer_id_2": 0.3}

        # Set up mock
        mock_get_name.side_effect = ["name1", "name2"]

        # Create the rewards message
        message = self.publisher._create_rewards_message(
            rewards_data, round_num, stage_num
        )

        # Check that the message is a RewardsMessage
        assert isinstance(message, RewardsMessage)

        # Check that the message has the correct type
        assert message.type == "rewards"

        # Check that the message has the correct data
        assert len(message.data) == 2

        # Check the first data item
        assert isinstance(message.data[0], RewardsMessageData)
        assert message.data[0].peer_id == "peer_id_1"
        assert message.data[0].peer_name == "name1"
        assert message.data[0].amount == 0.5
        assert message.data[0].round == round_num
        assert message.data[0].stage == stage_num

        # Check the second data item
        assert isinstance(message.data[1], RewardsMessageData)
        assert message.data[1].peer_id == "peer_id_2"
        assert message.data[1].peer_name == "name2"
        assert message.data[1].amount == 0.3
        assert message.data[1].round == round_num
        assert message.data[1].stage == stage_num


class TestGossipDHTPublisher:
    """Tests for the GossipDHTPublisher class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_dht = MagicMock()
        self.mock_kinesis = MagicMock()

        # Create a real logger for testing
        self.mock_logger = logging.getLogger("test_logger")
        self.mock_logger.setLevel(logging.INFO)

        # Add a handler to the logger so caplog can capture the logs
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Use the JSON formatter from python-json-logger
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(message)s %(extra)s"
        )
        handler.setFormatter(json_formatter)
        self.mock_logger.addHandler(handler)

        self.coordinator = MagicMock()

        # Create the publisher with a short poll interval for testing
        self.publisher = GossipDHTPublisher(
            dht=self.mock_dht,
            kinesis_client=self.mock_kinesis,
            logger=self.mock_logger,
            poll_interval_seconds=0.1,  # 100ms for faster tests
            coordinator=self.coordinator,
        )

    def teardown_method(self):
        """Clean up after tests."""
        # Stop the publisher if it's running
        if self.publisher.running:
            self.publisher.stop()
            # Give it a moment to stop
            time.sleep(0.2)

    def test_initialization(self, caplog):
        """Test that the publisher initializes correctly."""
        # Set the caplog level to capture INFO messages
        caplog.set_level(logging.INFO)

        # Re-initialize the publisher to capture the logs
        self.publisher = GossipDHTPublisher(
            dht=self.mock_dht,
            kinesis_client=self.mock_kinesis,
            logger=self.mock_logger,
            poll_interval_seconds=0.1,
            coordinator=self.coordinator,
        )

        # Verify the log message was captured
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "GossipDHTPublisher initialized"

        # Verify other properties
        assert self.publisher.dht == self.mock_dht
        assert self.publisher.kinesis_client == self.mock_kinesis
        assert self.publisher.coordinator == self.coordinator
        assert self.publisher.poll_interval_seconds == 0.1
        assert self.publisher.current_round == -1
        assert self.publisher.current_stage == -1
        assert self.publisher.last_polled is None
        assert self.publisher.running is False
        assert self.publisher._poll_thread is None

    def test_poll_once_no_rewards(self, caplog):
        """Test gossip polling when there's no rewards data."""

        # Set up mocks
        self.coordinator.get_round_and_stage.return_value = (1, 1)
        self.publisher._get_rewards_data = MagicMock(return_value=None)

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Error polling for round/stage in gossip"
        assert caplog.records[0].error == "missing rewards"

    def test_poll_once_with_rewards(self, caplog):
        """Test gossip polling when there is rewards data."""
        # Set the caplog level to capture INFO messages
        caplog.set_level(logging.INFO)

        # Set up mocks
        self.coordinator.get_round_and_stage.return_value = (1, 1)
        rewards_data = {"peer_id_1": 0.5, "peer_id_2": 0.3}
        self.publisher._get_rewards_data = MagicMock(return_value=rewards_data)

        # Mock the _get_outputs_data method to return None (no outputs)
        self.publisher._get_outputs_data = MagicMock(return_value=None)

        # Mock the _publish_gossip method
        self.mock_kinesis.put_gossip = MagicMock()

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that _get_rewards_data was called with the correct arguments
        self.publisher._get_rewards_data.assert_called_once_with(1, 1)

        # Check that _publish_gossip was called
        self.mock_kinesis.put_gossip.assert_not_called()

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Polled for round/stage"
        assert caplog.records[0].round == 1
        assert caplog.records[0].stage == 1

        assert caplog.records[1].message == "Publishing gossip messages"
        assert caplog.records[1].num_messages == 0

        # Check that last_polled was updated
        assert self.publisher.last_polled is not None

    def test_poll_once_error(self, caplog):
        """Test polling when there's an error."""
        # Set the caplog level to capture ERROR messages
        caplog.set_level(logging.ERROR)

        # Set up the coordinator mock to raise an exception
        self.coordinator.get_round_and_stage.side_effect = Exception("Test error")

        # Poll once
        self.publisher._poll_once()

        # Check that get_round_and_stage was called on the coordinator
        self.coordinator.get_round_and_stage.assert_called_once()

        # Check that the logger was called with the error
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Error polling for round/stage in gossip"
        assert caplog.records[0].error == "Test error"

        # Check that last_polled was not updated
        assert self.publisher.last_polled is None

    def test_publish_gossip(self, caplog):
        """Test publishing gossip data."""
        # Set the caplog level to capture INFO messages
        caplog.set_level(logging.INFO)

        # Set up test data
        gossip = [
            (
                1000.0,
                {
                    "id": "id1",
                    "message": "message1",
                    "node": "node1",
                    "nodeId": "peer_id_1",
                },
            ),
            (
                1001.0,
                {
                    "id": "id2",
                    "message": "message2",
                    "node": "node2",
                    "nodeId": "peer_id_2",
                },
            ),
        ]

        # Mock the put_gossip method
        self.mock_kinesis.put_gossip = MagicMock()

        # Publish gossip
        self.publisher._publish_gossip(gossip)

        # Check that put_gossip was called
        self.mock_kinesis.put_gossip.assert_called_once()

        # Check that the logger was called
        assert len(caplog.records) > 0
        assert caplog.records[0].message == "Publishing gossip messages"
        assert caplog.records[0].num_messages == 2

        assert caplog.records[1].message == "Successfully published gossip"
