# 定义模型预测函数
import paddle
import paddle.nn.functional
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

from config import config
from .data import convert_example


def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    # 将输入数据（list格式）处理为模型可接受的格式
    for text_a in data:
        input_ids, segment_ids = convert_example(
            {"text_a": text_a},
            tokenizer,
            max_seq_length=128,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = paddle.nn.functional.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results  # 返回预测结果


def do_test():
    params_path = f"./checkpoint/{config.dataset}/model_state.pdparams"
    data = [
        "河南工业大学",
        "郑州赛欧思科技有限公司",
        "郑州大学",
        "郑州市人民政府",
        "河南省公安厅",
        "郑州市公安局",
        "扶沟县人民政府"
    ]
    model = RobertaForSequenceClassification.from_pretrained(config.model_name, num_classes=config.num_classes)
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    model.set_dict(paddle.load(params_path))
    print(f"Loaded parameters from {params_path}")
    results = predict(model, data, tokenizer, config.label_map, batch_size=config.batch_size)
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
