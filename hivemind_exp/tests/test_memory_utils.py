from hivemind_exp.runner.memory_utils import (
    Quantization,
    parse_param_count,
    parse_quantization,
)


def test_parse_param_count():
    # Valid
    assert parse_param_count("1.5B") == 1_500_000_000
    assert parse_param_count("2M") == 2_000_000

    assert parse_param_count("1.5b") == 1_500_000_000
    assert parse_param_count("2m") == 2_000_000

    # Invalid
    assert parse_param_count("2K") == 0.0
    assert parse_param_count("bnb") == 0.0


def test_parse_quantization():
    # Valid
    assert parse_quantization("bnb-4bit") == Quantization._4BIT
    assert parse_quantization("-16bit") == Quantization._16BIT

    # Invalid
    assert parse_quantization("-12bit") == Quantization.NONE
    assert parse_quantization("rabbit") == Quantization.NONE
    assert parse_quantization("") == Quantization.NONE


def test_parse_combined():
    s = "Gensyn/Qwen2.5-32b-Instruct-bnb-4bit"
    assert parse_param_count(s) == 32_000_000_000
    assert parse_quantization(s) == Quantization._4BIT
