import os
import logging

from utils import load_dataset
from model import HyperParameters, BertSentSimCheckModelTrainer

logger = logging.getLogger("train model")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
MODEL_DIR = "model"

fh = logging.FileHandler(os.path.join(MODEL_DIR, "train.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

BERT_PRETRAINED_MODEL = "../pretrained_models/RoBERTa_zh_L12_PyTorch"

param_config = {
    "max_len": 256,
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "fp16": False,
    "fp16_opt_level": "O1",
    "max_grad_norm": 1.0,
    "warmup_steps": 0.1,
    "algorithm": "BertForSentencePairModel"
}

hyper_parameter = HyperParameters()
hyper_parameter.__dict__ = param_config

dataset_path = "./data/train"
test_input_path = None
trainer = BertSentSimCheckModelTrainer(
    dataset_path,
    BERT_PRETRAINED_MODEL,
    hyper_parameter,
    test_input_path
)

trainer.train(MODEL_DIR, kfold=1)