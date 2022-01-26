import numpy as np
import paddle
from paddlenlp.datasets import DatasetBuilder

from config import config


class Data(DatasetBuilder):
    SPLITS = {
        'train': 'train.csv',  # 训练集
        'dev': 'dev.csv',  # 验证集
        'test': 'test.csv'  # 测试集
    }

    def _get_data(self, mode: str, **kwargs):
        return self.SPLITS[mode]

    def _read(self, filename: str, *args):
        with open(f"./dataset/{config.dataset}/{filename}") as fp:
            for line in fp:
                line = line.strip()
                if line.isspace():
                    continue
                text_a, label = line.split(",")
                yield {'text_a': text_a, "label": label}

    def get_labels(self):
        return config.labels


# 定义数据集加载函数
def load_dataset(name=None, data_files=None, splits=None, lazy=None, **kwargs):
    reader_instance = Data(lazy=lazy, name=name, **kwargs)
    return reader_instance.read_datasets(data_files=data_files, splits=splits)


# 定义处理函数
def convert_example(example, tokenizer, max_seq_length=config.max_seq_length, is_test=False):
    qtconcat = example["text_a"]
    encoded_inputs = tokenizer(text=qtconcat, max_seq_len=max_seq_length)  # tokenizer处理为模型可接受的格式
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label


# 定义数据加载函数dataloader
def create_dataloader(dataset, mode='train', batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    # 训练数据集随机打乱，测试数据集不打乱
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=True)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=False)
    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
