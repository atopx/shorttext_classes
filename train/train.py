import os
import random
import time
from functools import partial

import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

from config import config
from . import data


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    accu = metric.accumulate()
    print("dev -- eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu  # 返回准确率


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def do_train():
    # paddle.set_device('gpu')
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(1000)
    train_ds, dev_ds = data.load_dataset(splits=["train", "dev"])

    # 通过在预训练模型后拼接上一个全连接网络（Full Connected）进行分类
    model = RobertaForSequenceClassification.from_pretrained(config.model_name, num_classes=config.num_classes)
    # 定义模型对应的tokenizer，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)

    trans_func = partial(data.convert_example, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack()  # labels
    ): [data for data in fn(samples)]

    # 训练集迭代器
    train_data_loader = data.create_dataloader(
        train_ds, mode='train', batch_size=config.batch_size,
        batchify_fn=batchify_fn, trans_fn=trans_func
    )

    # 验证集迭代器
    dev_data_loader = data.create_dataloader(
        dev_ds, mode='dev', batch_size=config.batch_size,
        batchify_fn=batchify_fn, trans_fn=trans_func
    )
    # model = paddle.DataParallel(model)
    num_training_steps = len(train_data_loader) * config.epochs
    lr_scheduler = ppnlp.transformers.LinearDecayWithWarmup(
        config.learning_rate, num_training_steps, config.warmup_proportion)

    # AdamW优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
    )
    criterion = paddle.nn.loss.CrossEntropyLoss()  # 交叉熵损失函数
    metric = paddle.metric.Accuracy()  # accuracy评价指标
    pre_accu = 0
    global_step = 0
    tic_train = time.time()
    print("start ...")
    for epoch in range(1, config.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = paddle.nn.functional.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()
            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                       10 / (time.time() - tic_train))
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            # 每10次输出日志
            if global_step % 10 == 0 and rank == 0:
                evaluate(model, criterion, metric, dev_data_loader)
            # 每100次验证一次
            if global_step % 100 == 0 and rank == 0:
                acc = evaluate(model, criterion, metric, dev_data_loader)
                if acc > pre_accu:  # 如果准确率高于上一次的就保存
                    if not os.path.exists(config.save_dir):
                        os.makedirs(config.save_dir)
                    model.save_pretrained(config.save_dir)
                    tokenizer.save_pretrained(config.save_dir)
                    pre_accu = acc
