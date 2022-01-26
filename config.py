class Config(object):
    # 训练配置参数
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.model_name = "roberta-wwm-ext"
        self.save_dir = f"./checkpoint/{self.dataset}"
        self.learning_rate = 4e-5  # 最大学习率
        self.batch_size = 128  # 批处理大小
        self.max_seq_length = 32  # 文本序列最大截断长度
        self.epochs = 3  # 训练轮次
        self.warmup_proportion = 0.1  # 学习率预热比例
        self.weight_decay = 0.01  # 权重衰减系数,类似模型正则项策略,避免模型过拟合
        self.labels = []
        with open(f"dataset/{self.dataset}/label.txt") as fp:
            for line in fp:
                line = line.strip()
                if line.isspace():
                    continue
                self.labels.append(line)
        self.label_map = dict(enumerate(self.labels))
        self.num_classes = len(self.labels)

    def __repr__(self):
        return "=======================[config]=======================\n" + \
               f"dataset: {self.dataset}\n" + \
               f"model_name: {self.model_name}\n" + \
               f"save_dir: {self.save_dir}\n" + \
               f"learning_rate: {self.learning_rate}\n" + \
               f"batch_size: {self.batch_size}\n" + \
               f"max_seq_length: {self.max_seq_length}\n" + \
               f"epochs: {self.epochs}\n" + \
               f"warmup_proportion: {self.warmup_proportion}\n" + \
               f"weight_decay: {self.weight_decay}\n" + \
               f"num_classes: {self.num_classes}\n" + \
               f"labels: {self.labels}\n" + \
               "=======================[config]======================="


config = None


def init_config(dataset):
    global config
    config = Config(dataset)
    print(config)
