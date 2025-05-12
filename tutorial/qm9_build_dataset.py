import argparse
from geqtrain.utils.torch_geometric.qm9 import QM9

def main():
    parser = argparse.ArgumentParser(description="Process the QM9 dataset.")
    parser.add_argument(
        "-d",
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the folder where the QM9 dataset will be stored."
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    qm9 = QM9(dataset_folder)
    qm9.download()
    qm9.process()

if __name__ == "__main__":
    main()