import argparse
from geqtrain.utils.torch_geometric.qm9 import QM9

def main(root):
    qm9 = QM9(root)
    qm9.download()
    qm9.process()
    qm9.split(split_on_key="mu", num_train_samples=110000, num_valid_samples=10000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and split the QM9 dataset.")
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        required=True,
        help="Root folder where the QM9 dataset will be saved."
    )
    args = parser.parse_args()
    main(args.root)