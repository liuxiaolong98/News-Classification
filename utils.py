import os
import json
import csv
import torch
import pandas as pd

from typing import List, Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler



class TextPairDataset(Dataset):
    """
    句子分类数据集
    """
    def __init__(self,
                 text_a_list: List[str],
                 text_b_list: List[str],
                 label_list: List[str]=None):
        if not label_list or len(label_list) == 0:
            label_list = [0] * len(text_a_list)
        assert all(
            len(label_list) == len(text_list)
            for text_list in [text_a_list, text_b_list]
        )
        self.text_a_list = text_a_list
        self.text_b_list = text_b_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text_a, text_b, label = (
            self.text_a_list[index],
            self.text_b_list[index],
            self.label_list[index]
        )
        return text_a, text_b, label

    @classmethod
    def from_dataframe(cls, df):
        text_a_list = df.iloc[:, 0].tolist()
        text_b_list = df.iloc[:, 1].tolist()
        labels = df.iloc[:, 2].tolist()
        # labels_int = []
        # for label in labels:
        #     if label.strip() == "负":
        #         labels_int.append(0)
        #     elif label.strip() == "非负":
        #         labels_int.append(1)
        #     else:
        #         labels_int.append(2)

        return cls(text_a_list, text_b_list, labels)

    @classmethod
    def from_dir(cls, data_dir):
        """
        :param data_dir:
        :return: 返回dataframe格式数据，三列分别为title，news，labels
        """
        # import re
        # pattern = re.compile("http.+")
        #
        # data_path = os.path.join(data_dir, "news_2552.xlsx")
        data_total = {}
        titles = []
        news = []
        labels = []
        nagetive_path = os.path.join(data_dir, "law_news/negative/3.txt")
        with open(nagetive_path, "r") as f:
            for line in f:
                data = json.loads(line)
                title = data["originalData"]["title"]
                summary = data["originalData"]["summary"]
                titles.append(title)
                news.append(summary)
                labels.append(0)

        unknown_path = os.path.join(data_dir, "law_news/unknown/1.txt")
        with open(unknown_path, "r") as f:
            for line in f:
                data = json.loads(line)
                title = data["originalData"]["title"]
                summary = data["originalData"]["summary"]
                titles.append(title)
                news.append(summary)
                labels.append(1)

        sensitive_path = os.path.join(data_dir, "law_news/sensitive/1.txt")
        with open(sensitive_path, "r") as f:
            for line in f:
                data = json.loads(line)
                title = data["originalData"]["title"]
                summary = data["originalData"]["summary"]
                titles.append(title)
                news.append(summary)
                labels.append(0)

        # sensitive_aug_path = os.path.join(data_dir, "law_news/sensitive/sensitive_augment.csv")
        # with open(sensitive_aug_path, "r") as f:
        #     f_csv = csv.reader(f)
        #     for line in f_csv:
        #         title = line[0]
        #         summary = line[1]
        #         titles.append(title)
        #         news.append(summary)
        #         labels.append(2)

        # if os.path.exists(data_path):
        #     df = pd.read_excel(data_path)
        #     titles = df["标题"].tolist()
        #     news_raw = df["发送预览"].tolist()
        #     labels = df["高级属性"].tolist()
            # with open("./data/train/augment_data.csv", "r") as f:
            #     f_csv = csv.reader(f)
            #     for line in f_csv:
            #         title = line[1]
            #         new = line[4]
            #         lable = line[5]
            #         titles.append(title)
            #         news_raw.append(new)
            #         labels.append(lable)

            # news = []
            # for item in news_raw:
            #     if pattern.findall(item):
            #         item = pattern.sub("", item)
            #     try:
            #         item = item.split(":", maxsplit=1)[1]
            #         news.append(item)
            #     except:
            #         news.append(item)

            # labels_int = []
            # for label in labels:
            #     if label.strip() == "负":
            #         labels_int.append(0)
            #     elif label.strip() == "非负":
            #         labels_int.append(1)
            #     else:
            #         labels_int.append(2)
        data_total["news"] = news
        data_total["titles"] = titles
        data_total["labels"] = labels
        return cls.from_dict_list(data_total)

    @classmethod
    def from_test(cls):
        id_path = "./data/test/id.xls"
        id_df = pd.read_excel(id_path)

        id_data = id_df["信息ID号"].to_list()[:100]
        id_set = set(id_data)

        data_total = {}
        news = []
        titles = []
        labels = []
        with open("./data/test/800.txt", "r") as f:
            for line in f:
                data_line = json.loads(line)
                id = data_line["id"]
                if id in id_set:
                    title = data_line["originalData"]["title"]
                    new = data_line["originalData"]["summary"]
                    news.append(new)
                    titles.append(title)
                    labels.append(0)

        df = pd.read_excel("./data/test/86.xls")
        new_title = df["标题"].to_list()
        new_summary = df["摘要"].to_list()
        new_label = df["算法属性"].to_list()
        new_labels = []
        for item in new_label:
            if item == "非负":
                new_labels.append(1)
            # elif item == "敏感":
            #     new_labels.append(2)
            else:
                new_labels.append(0)

        news.extend(new_summary)
        titles.extend(new_title)
        labels.extend(new_labels)

        data_total["news"] = news
        data_total["titles"] = titles
        data_total["labels"] = labels
        return cls.from_dict_list(data_total)


    @classmethod
    def from_dict_list(cls, data:List[str]):
        df = pd.DataFrame(data)
        return cls.from_dataframe(df)

def load_dataset(
        train_dataset_path,
        test_data_path=None,
        n_split: int=1
):
    train_data = TextPairDataset.from_dir(train_dataset_path)
    if test_data_path:
        test_data = TextPairDataset.from_dir(test_data_path)
        train_data.text_a_list.extend(test_data.text_a_list)
        train_data.text_b_list.extend(test_data.text_b_list)
        train_data.label_list.extend(test_data.label_list)

    text_a_list = train_data.text_a_list
    text_b_list = train_data.text_b_list
    label_list = train_data.label_list

    data_dict = {
        "text_a_list": text_a_list,
        "text_b_list": text_b_list,
        "label_list": label_list
    }
    # data_dict = {"text_a_list": train_data.text_a_list,
    #              "text_b_list": train_data.text_b_list,
    #              "label_list": train_data.label_list}
    df = pd.DataFrame(data_dict)

    data = []
    if n_split == 1:
        df = shuffle(df, random_state=42)
        total_len = len(df)
        train_len = int(total_len*0.7)
        train_df = df[:train_len]
        test_df = df[train_len:]

        training_data = TextPairDataset.from_dataframe(train_df)
        testing_data = TextPairDataset.from_dataframe(test_df)
        data.append((training_data, testing_data))
        return data
    else:
        kf = StratifiedKFold(n_split, shuffle=True)
        for train_index, test_index in kf.split(df, df.iloc[:, 2]):
            train_df, test_df = (
                df.iloc[train_index].reset_index(drop=True),
                df.iloc[test_index].reset_index(drop=True)
            )
            train_data = TextPairDataset.from_dataframe(train_df)
            test_data = TextPairDataset.from_dataframe(test_df)
            data.append((train_data, test_data))
        return data

class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
    def to_tensor(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.LongTensor(self.input_ids).to(device),
            torch.LongTensor(self.segment_ids).to(device),
            torch.LongTensor(self.input_mask).to(device),
        )

class InputExample:
    def __init__(self, text_a, text_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    @staticmethod
    def _text_pair_to_feature(text_a, text_b, tokenizer, max_len):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        if text_b:
            tokens_b = tokenizer.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_len-3)
        else:
            if len(tokens_a) > max_len-2:
                tokens_a = tokens_a[: (max_len-2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        return input_ids, segment_ids, input_mask

    def to_two_pair_feature(self, tokenizer, max_seq_length) -> Tuple[InputFeatures, InputFeatures]:
        try:
            a = self._text_pair_to_feature(self.text_a, None, tokenizer, max_seq_length)
            b = self._text_pair_to_feature(self.text_b, None, tokenizer, max_seq_length)
            a, b = InputFeatures(*a), InputFeatures(*b)
            return a, b
        except:
            print('a', self.text_a)
            print('b', self.text_b)

    def to_single_text_feature(self, tokenizer, max_seq_length) -> InputFeatures:
        try:
            ab = self._text_pair_to_feature(self.text_a, self.text_b, tokenizer, max_seq_length)
            ab = InputFeatures(*ab)
            return ab

        except:
            print('a', self.text_a)
            print('b', self.text_b)


def _truncate_seq_pair(tokens_a: list, tokens_b: list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_collator(max_len, device, tokenizer, model_class):
    def single_collate_fn(batch):
        """
        获取一个mini batch的数据，将文本转化成tensor。
        将a、b拼接并编码为tensor
        :param batch:
        :return:
        """
        example_tensors = []
        for text_a, text_b, label in batch:
            input_example = InputExample(text_a, text_b, label)
            ab_feature = input_example.to_single_text_feature(tokenizer, max_len)
            ab_tensor = ab_feature.to_tensor(device)
            # label_tensor = torch.FloatTensor([label]).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            example_tensors.append((ab_tensor, label_tensor))
        return default_collate(example_tensors)

    def siamese_collate_fn(batch):
        """
        获取一个mini batch的数据，将文本转化成tensor。
        将a、b单独编码为tensor
        :param batch:
        :return:
        """
        example_tensors = []
        for text_a, text_b, label in batch:
            input_example = InputExample(text_a=text_b, label=label)
            b_feature = input_example.to_single_text_feature(tokenizer, max_len)
            b_tensor = b_feature.to_tensor(device)
            label_tensor = torch.LongTensor([label]).to(device)
            example_tensors.append((b_tensor, label_tensor))

        return default_collate(example_tensors)

    if model_class == "BertForSentencePairModel":
        return single_collate_fn
        # return no_tagging_collate_fn
    elif model_class == "BertForSentencePairModelV2":
        return siamese_collate_fn

def loader(
        data,
        max_len,
        batch_size,
        device,
        tokenizer,
        model_class,
        mode
)->None:
    if mode == "train":
        train_sample = RandomSampler(data)
        collate_fn = get_collator(max_len,
                                  device,
                                  tokenizer,
                                  model_class)

        train_dataloader = DataLoader(dataset=data,
                                      sampler=train_sample,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=collate_fn,
                                      drop_last=True)
        return train_dataloader

    elif mode == "predict":
        sampler = SequentialSampler(data)
        collate_fn = get_collator(max_len,
                                  device,
                                  tokenizer,
                                  model_class)

        data_loader = DataLoader(dataset=data,
                                 sampler=sampler,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)
        return data_loader



if __name__=="__main__":
    print("*******************")
    pretrained_model_path = "../pretrain/RoBERTa_zh_L12_PyTorch"
    train_dataset_path = "./data/train"
    train_data = TextPairDataset.from_dir(train_dataset_path)

    from collections import Counter

    label_list = Counter(train_data.label_list)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(train_dataset_path, None, 1)[0][0]

    data_loader = loader(data, 256, 8, device, tokenizer, "BertForSentencePairModel", "train")
    for batch in data_loader:
        print(batch)

    print("hello")