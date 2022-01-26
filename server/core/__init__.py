from core.predictor import Predictor
from core.config import Config


def init_models(targets) -> dict:
    import json
    with open("metadata.json") as fp:
        metadata = json.load(fp)
    return {t: Predictor(Config(metadata[t])) for t in targets}
