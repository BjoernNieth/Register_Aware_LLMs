import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--mfte-tagger-path', type=str)
    parser.add_argument('--huggingface-token', type=str)
    parser.add_argument('--config-file-path', type=str)

    return parser
