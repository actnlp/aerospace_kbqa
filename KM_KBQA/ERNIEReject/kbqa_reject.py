from .torch_utils import load_config
from .loader_foronetext import DataLoader
from .trainer import MyTrainer
from ..config import config
import requests, os

id2label = {0: 'kbqa', 1: 'cqa'}
best_model_path = config.ernie_best_model_path  # load model
print("best_model_path ", best_model_path)
opt = load_config(best_model_path)
opt["cuda"] = False
cls_model = MyTrainer(opt)
cls_model.load(best_model_path)


def query_type_infer(text):
    # data encode
    data = [{"text_a": text, "label": ""}]
    test_batch = DataLoader(data, opt['batch_size'], opt)

    item = test_batch[0]
    top_n = 1
    infer_res = cls_model.infer(item, top_n + 1)
    labels_list = list(infer_res.keys())
    return labels_list[0]
    # labels = [id2label[labels_list[i]] for i in range(0, top_n + 1)]
    # labels = [labels_list[i] for i in range(0, top_n + 1)]
    # return labels[:top_n]
