import os

import paddle
from paddlenlp.transformers import RobertaTokenizer


class Config(object):
    def __init__(self, data: dict):
        self.device = os.getenv("device", "cpu")
        self.cpu_threads = os.getenv("threads", 10)
        self.max_seq_length = 32
        self.batch_size = 128
        self.use_tensorrt = False
        self.model_name = "roberta-wwm-ext"
        self.label_map = dict(enumerate(data["labels"]))
        self.model_file = data["model_file"]
        self.params_file = data["params_file"]

        if not os.path.exists(self.model_file):
            raise ValueError(f"not find model file path {self.model_file}")

        if not os.path.exists(self.params_file):
            raise ValueError(f"not find params file path {self.params_file}")

        self.interface = paddle.inference.Config(self.model_file, self.params_file)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
