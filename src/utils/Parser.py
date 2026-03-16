import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--huggingface-token', type=str)
    parser.add_argument('--config-file-path', type=str)
    parser.add_argument('--test-mode', type=bool, default=False, help="Run in test mode with a small dataset sample")

    return parser