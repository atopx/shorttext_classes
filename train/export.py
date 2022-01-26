import os

import paddle
from paddlenlp.transformers import RobertaForSequenceClassification

from config import config


def do_export():
    model = RobertaForSequenceClassification.from_pretrained(config.model_name, num_classes=config.num_classes)
    state_dict = paddle.load(f'{config.save_dir}/model_state.pdparams')
    model.set_dict(state_dict)
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64")  # segment_ids
        ])
    paddle.jit.save(model, os.path.join(f"./models/{config.dataset}/inference"))
