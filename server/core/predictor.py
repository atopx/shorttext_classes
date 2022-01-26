import numpy as np
import paddle
from paddlenlp.data import Tuple, Pad
from scipy.special import softmax


class Predictor(object):
    def __init__(self, config):
        self.config = config
        if self.config.device == "cpu":
            self.config.interface.disable_gpu()
            self.config.interface.set_cpu_math_library_num_threads(config.cpu_threads)
        elif config.device.upper() == "gpu":
            self.config.interface.enable_use_gpu(100, 0)
            precision_mode = paddle.inference.PrecisionType.Float32
            if config.use_tensorrt:
                config.interface.enable_tensorrt_engine(
                    max_batch_size=config.batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode
                )
        elif config.device == "xpu":
            config.interface.enable_xpu(100)
        else:
            raise ValueError(f"unsupported device {config.device}")

        config.interface.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config.interface)
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.batchify_fn = lambda x, fn=Tuple(
            Pad(axis=0, pad_val=self.config.tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=self.config.tokenizer.pad_token_id),  # segment
        ): fn(x)

    # 转换函数
    async def convert(self, source: str) -> tuple:
        # tokenizer处理为模型可接受的格式
        encoded_inputs = self.config.tokenizer(text=source, max_seq_len=self.config.max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        return input_ids, token_type_ids

    async def predict(self, source: str):
        input_ids, token_type_ids = self.batchify_fn([await self.convert(source)])
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(token_type_ids)
        self.predictor.run()
        idx = np.argmax(softmax(self.output_handle.copy_to_cpu(), axis=1), axis=1).tolist()
        return self.config.label_map[idx[0]]

    async def batch_predict(self, record: list) -> dict:
        input_ids, token_type_ids = self.batchify_fn([await self.convert(item) for item in record])
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(token_type_ids)
        self.predictor.run()
        idx = np.argmax(softmax(self.output_handle.copy_to_cpu(), axis=1), axis=1).tolist()
        return dict(zip(record, [self.config.label_map[i] for i in idx]))
