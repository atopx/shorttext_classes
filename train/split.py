from config import config
from sklearn.model_selection import train_test_split


def do_split():
    # 加载原始数据
    x = []
    y = []
    file = open(f"dataset/{config.dataset}/source.csv")
    for line in file:
        line = line.strip()
        if line.isspace():
            continue
        value, label = line.split(",")
        x.append(value)
        y.append(label)

    assert len(x) == len(y)  # 验证数据加载

    # 测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)
    with open(f"dataset/{config.dataset}/test.csv", "w") as fp:
        for v, t in zip(x_test, y_test):
            fp.write(f"{v},{t}\n")

    # 验证集
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1)
    with open(f"dataset/{config.dataset}/dev.csv", "w") as fp:
        for v, t in zip(x_dev, y_dev):
            fp.write(f"{v},{t}\n")

    # 训练集
    with open(f"dataset/{config.dataset}/train.csv", "w") as fp:
        for v, t in zip(x_train, y_train):
            fp.write(f"{v},{t}\n")
