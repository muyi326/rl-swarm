import argparse
import tempfile

import yaml


def output_file(args, yaml_header, param_b, name_suffix=""):
    if param_b == int(param_b):
        param_b = int(param_b)

    model_name = f"{args.model_prefix}-{param_b}B-Instruct{name_suffix}"
    model_base = model_name.rpartition("/")[-1]
    output_yaml = f"{args.yaml_output_dir}/{args.yaml_prefix}-{param_b}b{name_suffix}-{args.yaml_suffix}.yaml"

    extra = {
        "model_name_or_path": model_name,
        "output_dir": f"runs/gsm8k/multinode/{model_base}-Gensyn-Swarm",
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tf:
        tf.write("\n# Model-specific arguments\n")
        yaml.dump(extra, tf)

        with open(output_yaml, "w") as of:
            print(f"Writing to {output_yaml}")
            for filename in (yaml_header, tf.name):
                with open(filename, "r") as infile:
                    of.write(infile.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML configuration files.")
    parser.add_argument(
        "--yaml_prefix", type=str, required=True, help="Prefix for output YAML files."
    )
    parser.add_argument(
        "--yaml_suffix", type=str, required=True, help="Suffix for output YAML files."
    )
    parser.add_argument(
        "--yaml_header", type=str, required=True, help="Path to YAML header file."
    )
    parser.add_argument(
        "--yaml_big_header", type=str, required=True, help="Path to big YAML header file."
    )
    parser.add_argument(
        "--yaml_output_dir", type=str, help="Directory to write output YAML files."
    )

    parser.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="Model name prefix. E.g. Gensyn-7B.",
    )
    parser.add_argument(
        "--param_counts",
        type=float,
        nargs="+",
        required=True,
        help="List of parameter counts (in B).",
    )
    parser.add_argument(
        "--param_counts_4bit",
        type=float,
        default=[],
        nargs="+",
        help="List of parameter counts (in B) to use unsloth's dynamic 4-bit quantization with.",
    )

    args = parser.parse_args()
    for p in args.param_counts:
        output_file(args, args.yaml_header, p)

    for p in args.param_counts_4bit:
        output_file(args, args.yaml_big_header, p, "-bnb-4bit")
