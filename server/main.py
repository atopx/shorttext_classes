from fastapi import FastAPI, Request

import core

app = FastAPI()

# 初始化预测模型
datasets = ["nature", "area", "industry"]
handlers = core.init_models(datasets)


@app.get("/ping")
def pint() -> dict:
    return {"message": "pong", "status": True}


@app.get("/predict/multi")
async def predict_multi(source: str) -> dict:
    # 多属性预测方法
    result = {"status": True}
    for dataset in datasets:
        value = await handlers[dataset].predict(source)
        result.setdefault(dataset, value)
    return result


@app.get("/predict/{dataset}")
async def predict_single(dataset: str, source: str) -> dict:
    # 单属性预测方法
    if dataset not in datasets:
        return {"status": False, "message": "Invalid attribute"}
    value = await handlers[dataset].predict(source)
    return {dataset: value, "status": True}


@app.post("/predict/batch/multi")
async def batch_predict_multi(request: Request) -> dict:
    # 批量多属性预测方法
    result = {"status": True}
    args = await request.json()
    if args.get("record") is None:
        result["data"] = {}
    else:
        for dataset in datasets:
            value = await handlers[dataset].batch_predict(args["record"])
            result.setdefault(dataset, value)
    return result


@app.post("/predict/batch/{dataset}")
async def batch_predict_single(request: Request, dataset: str) -> dict:
    # 批量单属性预测方法
    if dataset not in datasets:
        return {"status": False, "message": "Invalid attribute"}
    args = await request.json()
    value = await handlers[dataset].batch_predict(args.get("record", []))
    return {dataset: value, "status": True}
