import os
import json
import torch
import logging
import numpy as np
import random
import torch.nn.functional as F

from tqdm import tqdm
from typing import List, Union, Tuple
from sklearn.metrics import classification_report, accuracy_score
from torch.nn import CrossEntropyLoss, BCELoss
from transformers import (BertPreTrainedModel,
                          BertModel,
                          BertTokenizer,
                          WEIGHTS_NAME,
                          CONFIG_NAME,
                          get_linear_schedule_with_warmup,
                          AdamW,
                          BertConfig)
from utils import TextPairDataset, load_dataset, loader
logger = logging.getLogger("train model")

class HyperParameters:
    """
    用于管理模型超参数
    """
    def __init__(self,
                 max_len: int=512,
                 epochs: int=5,
                 batch_size: int=16,
                 learning_rate=2e-5,
                 fp16=True,
                 fp16_opt_level="01",
                 max_grad_norm=1.0,
                 warmup_steps=0.1,
                 algorithm="BertForSentencePairModel")->None:
        self.max_len = max_len
        """句子的最大长度"""
        self.epochs = epochs
        """训练迭代轮数"""
        self.batch_size = batch_size
        """每个batch的样本数量"""
        self.learning_rate = learning_rate
        """学习率"""
        self.fp16 = fp16
        """是否使用fp16混合精度训练"""
        self.fp16_opt_level = fp16_opt_level
        """用于fp16，Apex AMP优化等级，['O0', 'O1', 'O2', and 'O3']可选，详见https://nvidia.github.io/apex/amp.html"""
        self.max_grad_norm = max_grad_norm
        """最大梯度裁剪"""
        self.warmup_steps = warmup_steps
        """学习率线性预热步数"""
        self.algorithm = algorithm

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    @classmethod
    def from_json_file(cls, param_path: str):
        hyperp = cls()
        if os.path.exists(param_path):
            hyperp_config = json.load(open(param_path))
            hyperp.__dict__.update(hyperp_config)
        return hyperp

# class BertForSentencePairModel(BertPreTrainedModel):
#     """
#     a,b交互编码
#     """
#     def __init__(self, config, *input, **kwargs):
#         super().__init__(config, *input, **kwargs)
#         self.bert = BertModel(config)
#         self.num_lables = 2
#         self.seq_relationship = torch.nn.Linear(config.hidden_size, self.num_lables)
#         # self.dropout= torch.nn.Dropout(config.attention_probs_dropout_prob)
#         self.activate = torch.nn.Softmax(dim=1)
#         self.init_weights()
#
#     def forward(self, ab, labels=None, mode="prob"):
#         ab_pooled_out = self.bert(*ab)[1]
#         #dropout
#         # ab_pooled_out = self.dropout(ab_pooled_out)
#         output = self.seq_relationship(ab_pooled_out)
#         output = self.activate(output)
#
#         if mode == "prob":
#             return output
#         elif mode=="loss":
#             loss = F.cross_entropy(output.view(-1, self.num_lables), labels.view(-1))
#             return loss
#         elif mode=="evaluate":
#             loss = F.cross_entropy(output.view(-1, self.num_lables), labels.view(-1))
#             return output, loss

# class BertForSentencePairModel(BertPreTrainedModel):
#     """
#     a、b 交互并编码
#     """
#     def __init__(self, config, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.bert = BertModel(config)
#         # self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = torch.nn.Dropout(0.5)
#         self.seq_relationship = torch.nn.Linear(config.hidden_size, 1)
#         self.init_weights()
#
#     def FocalLoss(self, inputs, targets):
#         alpha = 1
#         gamma = 2
#         logits = False
#         reduce = True
#         if logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#
#         pt = torch.exp_(-BCE_loss)
#         F_loss = alpha * (1 - pt)**gamma * BCE_loss
#         if reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
#
#     def forward(self, ab, labels=None, mode="prob"):
#         ab_pooled_output = self.bert(*ab)[1]
#         # ab_pooled_output = self.LayerNorm(ab_pooled_output)
#         dropout_output = self.dropout(ab_pooled_output)
#         output = self.seq_relationship(dropout_output)
#         output = torch.sigmoid(output)
#
#         if mode == "prob" or mode == "logits":
#             return output
#         elif mode == "loss":
#             # loss_fn = BCELoss(weight = torch.FloatTensor([1 if i==1 else 5 for i in labels]).to('cuda'))
#             # loss = loss_fn(output.view(-1), labels.view(-1))
#             # return loss
#             loss = self.FocalLoss(output.view(-1), labels.view(-1))
#             return loss
#         elif mode=="evaluate":
#             # loss_fn = BCELoss(weight=torch.FloatTensor([1 if i == 1 else 5 for i in labels]).to('cuda'))
#             # loss = loss_fn(output.view(-1), labels.view(-1))
#             # return output, loss
#             loss = self.FocalLoss(output.view(-1), labels.view(-1))
#             return output, loss
#

class BertForSentencePairModel(BertPreTrainedModel):
    """
    a、b 交互并编码
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        # self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(0.3)
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)
        self.loss = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def logits_adjust_loss(self, y_true, y_pred, tau=1.0):
        prior = torch.tensor([0.2, 0.8], device="cuda")
        log_prior = torch.log(prior + 1e-2)
        log_prior = log_prior.expand(y_pred.shape)
        y_pred = y_pred + tau * log_prior
        output = self.loss(y_pred.view(-1, 2), y_true.view(-1))
        return output

    def forward(self, ab, labels=None, mode="prob"):
        ab_pooled_output = self.bert(*ab)[1]
        # ab_pooled_output = self.LayerNorm(ab_pooled_output)
        dropout_output = self.dropout(ab_pooled_output)
        output = self.seq_relationship(dropout_output)
        # output = torch.sigmoid(output)

        if mode == "prob" or mode == "logits":
            return output[:, 0]
        elif mode == "loss":
            loss = self.logits_adjust_loss(labels, output)
            return loss
        elif mode=="evaluate":
            loss = self.logits_adjust_loss(labels, output)
            return output[:, 0], loss


class BertForSentencePairModelV2(BertPreTrainedModel):
    """
    a、b 单独编码并交互
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.sim = torch.nn.PairwiseDistance(keepdim=True)
        self.seq_relationship = torch.nn.Linear(1, 1)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, a, b, labels=None, mode="prob"):
        _, a_pooled_output = self.bert(*a, output_all_encoded_layers=False)
        _, b_pooled_output = self.bert(*b, output_all_encoded_layers=False)

        sim_ab = self.sim(a_pooled_output, b_pooled_output)

        output = self.seq_relationship(sim_ab)
        # output = torch.sigmoid(output)

        if mode == "prob":
            return output
        elif mode == "logits":
            return sim_ab
        elif mode == "loss":
            # print(labels)
            # weight = torch.FloatTensor([0.8] * len(labels)).to('cuda')
            loss_fn = torch.nn.BCELoss(weight = torch.FloatTensor([[5] if i==1 else [1] for i in labels]).to('cuda'))
            loss = loss_fn(output.view(-1), labels.view(-1))
            return loss

algorithm_map = {
    "BertForSentencePairModel": BertForSentencePairModel,
    "BertForSentencePairModelV2": BertForSentencePairModelV2,
}

class BertSentSimCheckModel:
    def __init__(self,
                 model,
                 tokenizer,
                 device: torch.device=None)->None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = HyperParameters()
        self.max_len = self.config.max_len
        self.algorithm = self.config.algorithm
        self.model_class = algorithm_map[self.algorithm]
        self.batch_size = self.config.batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        elif device == "gpu":
            self.device = torch.device("cuda")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_dir, device:str=None):
        """
        加载模型
        :param model_dir: 模型目录
        :param device: 使用什么计算设备进行推断，为 'cpu' 或者 'gpu' 中的一个。不选则优先使用 gpu，如无再使用 cpu
        :return:
        """
        with open(os.path.join(model_dir, "param.json")) as f:
            param_dict = json.load(f)
            algorithm = param_dict["algorithm"]
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
        model_class = algorithm_map[algorithm]
        model = model_class.from_pretrained(model_dir)
        return cls(model, tokenizer, device)

    def save(self, model_dir):
        # Only save the model it-self
        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(model_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_pretrained(model_dir)

        # Save parameters
        with open(os.path.join(model_dir, "param.json"), mode="w") as f:
            json.dump({"algorithm": self.algorithm}, f)

    def predict(self, data):
        data_loader = loader(data,
                             self.max_len,
                             self.batch_size,
                             self.device,
                             self.tokenizer,
                             self.algorithm,
                             mode="predict")
        final_results = []
        total_loss = []
        with torch.no_grad():
            steps = tqdm(data_loader)
        #     for i, batch in enumerate(steps, start=1):
        #         output, loss = self.model(*batch, mode="evaluate")
        #         predict_labels= output.max(dim=1)[1]
        #         predict_labels = predict_labels.cpu().detach().numpy()
        #
        #         final_result.extend(list(zip(predict_labels, output)))
        #         total_loss.append(loss.item())
        #         steps.set_description("Loss {:.5f}".format(loss.item()))
        #
        # return final_result, sum(total_loss)/len(total_loss)
            for i, batch in enumerate(steps, start=1):
                output, loss = self.model(*batch, mode="evaluate")
                # predict_results = output.cpu().detach().numpy()[:, 0]
                predict_results = output.cpu().detach().numpy()
                # print(predict_results)
                predict_labels = [int(i < 0.5) for i in predict_results]
                # print(predict_labels)
                final_results.extend(list(zip(predict_labels, predict_results)))
                total_loss.append(loss.item())
                steps.set_description("Loss {:.5f}".format(loss.item()))

        return final_results, sum(total_loss) / len(total_loss)


class BertSentSimCheckModelTrainer:
    def __init__(self,
                 dataset_path,
                 bert_model_dir,
                 param: HyperParameters,
                 test_input_path):
        """
        :param dataset_path: 训练集路径，默认当作是训练集，但当train函数采用了kfold参数时，将对该数据集进行划分并做交叉验证
        :param bert_model_dir: 预训练 bert 模型路径
        :param param: 超参数
        :param test_input_path: 测试集的路径，用于快速测试模型性能和可用性
        """
        self.dataset_path = dataset_path
        self.bert_model_dir = bert_model_dir
        self.param = param
        self.test_input_path = test_input_path
        self.algorithm = self.param.algorithm
        self.model_class = algorithm_map[self.algorithm]

    def train(self, model_dir, kfold=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        logger.info("***** Running training *****")
        logger.info("dataset: {}".format(self.dataset_path))
        logger.info("k-fold number: {}".format(kfold))
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
        logger.info(
            "config: {}".format(
                json.dumps(self.param.__dict__, indent=4, sort_keys=True)
            )
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        if n_gpu > 0:
            torch.cuda.manual_seed_all(42)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=True)

        data = load_dataset(self.dataset_path,
                            self.test_input_path,
                            n_split=kfold)

        all_acc_list = []
        all_loss_list = []
        for k, (train_data, test_data) in enumerate(data, start=1):
            one_fold_acc_list = []
            one_fold_loss_list = []
            bert_model = self.model_class.from_pretrained(self.bert_model_dir)

            # """冻结部分层的参数"""
            # logger.info("we are freazing the parameters of front {} layers".format(8))
            # for i in range(8):
            #     params = bert_model.bert.encoder.layer[i].parameters()
            #     for param in params:
            #         param.requires_grad = False

            bert_model.to(device)

            train_dataloader = loader(train_data,
                                  self.param.max_len,
                                  self.param.batch_size,
                                  device,
                                  tokenizer,
                                  self.algorithm,
                                  mode="train")

            num_train_optimization_steps = (int(len(train_data) / self.param.batch_size) * self.param.epochs)
            param_optimizer = list(bert_model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.01, },
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, },
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.param.learning_rate, eps=1e-8)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_train_optimization_steps * 0.1,
                num_training_steps=num_train_optimization_steps,
            )

            # if self.param.fp16:
            #     from apex import amp
            #     bert_model, optimizer = amp.initialize(
            #         bert_model, optimizer, opt_level=self.param.fp16_opt_level
            #     )

            if n_gpu > 1:
                bert_model = torch.nn.DataParallel(bert_model)

            bert_model.zero_grad()

            from collections import Counter

            label_num = dict(Counter(train_data.label_list))

            logger.info("***** fold {}/{} *****".format(k, kfold))
            logger.info("  Num examples = %d", len(train_data))
            logger.info("Num of Negative Samples = %d", label_num[0])
            logger.info("Num of Unknown Samples = %d", label_num[1])
            # logger.info("Num of Seneitive Samples = %d", label_num[2])
            logger.info("  Batch size = %d", self.param.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            bert_model.train()
            dev_best_loss = float("inf")
            total_batch = 1

            for epoch in range(int(self.param.epochs)):
                tr_loss = 0
                steps = tqdm(train_dataloader)
                for step, batch in enumerate(steps, start=1):
                    loss = bert_model(*batch, mode="loss")
                    if n_gpu > 1:
                        loss = loss.mean()
                    # if self.param.fp16:
                    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.param.max_grad_norm)
                    # else:
                    #     loss.backward()
                    #     torch.nn.utils.clip_grad_norm_(bert_model.parameters(), self.param.max_grad_norm)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bert_model.parameters(), self.param.max_grad_norm)

                    tr_loss += loss.item()
                    optimizer.step()
                    scheduler.step()
                    bert_model.zero_grad()
                    total_batch += 1
                    steps.set_description(
                        "Fold: {}/{}, Epoch: {}/{}, Loss: {:.5f}".format(k, kfold, epoch+1, self.param.epochs, loss.item())
                    )

                model = BertSentSimCheckModel(bert_model, tokenizer)
                acc, dev_loss = self.evaluate(model, test_data)
                one_fold_acc_list.append(acc)
                one_fold_loss_list.append(dev_loss)

                logger.info(
                    "Epoch: {}/{}, train_loss: {:.5f}, dev acc: {:.5f}, dev_loss: {:.5f}".format(
                        epoch+1, self.param.epochs, tr_loss, acc, dev_loss
                    )
                )

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    model.save(model_dir)
                bert_model.train()

            all_acc_list.append(one_fold_acc_list)
            all_loss_list.append(one_fold_loss_list)

        logger.info("***** Stats *****")
        # 计算kfold的平均的acc
        all_epoch_acc = list(zip(*all_acc_list))
        logger.info("acc for each epoch:")
        for epoch, acc in enumerate(all_epoch_acc, start=1):
            logger.info(
                "epoch: %d, mean: %.5f"
                % (epoch, float(np.mean(acc)))
            )
        logger.info("loss for each epoch:")
        for epoch, loss in enumerate(all_loss_list, start=1):
            logger.info(
                "epoch: %d, mean: %.5f" % (epoch, float(np.mean(loss)))
            )
        logger.info("***** Training complete *****")

    @staticmethod
    def evaluate(model: BertSentSimCheckModel,
                 data: TextPairDataset):
        """
        评估模型，计算acc
        :param model:
        :param data:
        :param real_label_list:
        :return:
        """
        num_padding = 0
        if isinstance(model.model, torch.nn.DataParallel):
            num_padding = (model.batch_size - len(data) % model.batch_size)
            if num_padding != 0:
                padding_data = TextPairDataset(text_a_list=[""]*num_padding,
                                               text_b_list=[""]*num_padding)
                data = data.__add__(padding_data)

        real_label_list = data.label_list
        predict_result, loss = model.predict(data)
        if num_padding != 0:
            predict_result = predict_result[:num_padding]
            real_label_list = real_label_list[:num_padding]

        try:
            assert len(predict_result) == len(real_label_list)
        except:
            print(len(predict_result), len(real_label_list))

        predict_result_without_prob = [item[0] for item in predict_result]

        # print(predict_result_without_prob)
        # print(real_label_list)

        logger.info("预测类别" + str(set(predict_result_without_prob)))
        logger.info("真实类别" + str(set(real_label_list)))
        logger.info("\n" + classification_report(real_label_list,
                                                 predict_result_without_prob,
                                                 target_names=["negative", "unknown"], digits=4))


        acc = accuracy_score(real_label_list, predict_result_without_prob)

        return acc, loss


if __name__=="__main__":
    print("*******")
    MODEL_DIR = "model"
    if os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    BERT_PRETRAINED_MODEL = "../pretrained/RoBERTa_zh_L12_PyTorch"

    param_config = {
        "max_len": 256,
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "fp16":False,
        "fp16_opt_level": "O1",
        "max_grad_norm": 1.0,
        "warmup_steps": 0.1,
        "algorithm": "BertForSentencePairModel"
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = param_config
    print(hyper_parameter)

    dataset_path = "./data/train"
    test_input_path = None
    trainer = BertSentSimCheckModelTrainer(
        dataset_path,
        BERT_PRETRAINED_MODEL,
        hyper_parameter,
        test_input_path
    )

    tmp = trainer.train(MODEL_DIR)
    for batch in tqdm(tmp):
        print("|")
