import argparse

from config import init_config

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, help="目标数据集", required=True)
parser.add_argument("--option", type=str, default="train", help="操作选项",
                    choices=["split", "train", "export", "test"], required=True)

args = parser.parse_args()


def main():
    init_config(args.dataset)

    if args.option == "split":
        from train.split import do_split
        do_split()

    elif args.option == "train":
        from train.train import do_train
        do_train()

    elif args.option == "test":
        from train.test import do_test
        do_test()

    else:
        from train.export import do_export
        do_export()


if __name__ == '__main__':
    main()
