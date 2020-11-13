import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--test_file", type=bool, default=True)

args = parser.parse_args()

if __name__ == "__main__":
    print("Inference start...")

    inference(
        folds=args.folds,
        task=args.task,
        model=args.model,
        metric=args.metric
    )