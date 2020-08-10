import argparse

def create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", type=str, default="RESNET18")
    parser.add_argument("--metric", type=str, default="ACCURACY")

    args = parser.parse_args()
    return args