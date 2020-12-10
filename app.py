import argparse
import json
import logging
import os

from model import BertSentSimCheckModel
from utils import TextPairDataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, make_response
from flask.logging import default_handler
from flask_cors import CORS
from werkzeug.exceptions import BadRequestKeyError

app = Flask(__name__)
CORS(app)


class StandardResponse(Exception):
    status_code = 200

    def __init__(
        self, is_success: bool = True, msg=None, status_code: int = 200, data=None
    ):
        Exception.__init__(self)
        if status_code is not None:
            self.status_code = status_code
        self.msg = msg
        self.data = data
        self.is_success = is_success

    def to_dict(self):
        rv = dict()
        rv["code"] = 0 if self.is_success else 1
        rv["data"] = self.data
        rv["msg"] = self.msg
        return rv


@app.errorhandler(StandardResponse)
def handle_standard_response(resp):
    resp_dict = resp.to_dict()
    result_json = json.dumps(resp_dict, ensure_ascii=False)
    response = make_response(result_json)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.status_code = resp.status_code
    return response


@app.route("/np", methods=["GET"])
def np():
    """
    文本正负面分类。
    :return:
    """
    title = clean_text(request.args.get("title", ""))
    summary = clean_text(request.args.get("summary", ""))
    # client = request.args.get("client", "")
    # text = title + " " + summary
    client = ""
    try:
        text = summary[summary.index(":") + 1 :]
    except:
        text = summary
    data_raw = {}
    data_raw["titles"] = [title]
    data_raw["news"] = [summary]
    data_raw["labels"] = [0]

    use_streamer = request.args.get("use_streamer", "false") == "true"

    data = TextPairDataset.from_dict_list(data_raw)
    try:
        # if use_streamer:
        #     predict_results, _ = streamer.predict(data)
        # else:
        #     predict_results, _ = model.predict(data)

        if use_streamer:
            predict_results= streamer.predict(data)
        else:
            predict_results, _= model.predict(data)

        result = [
            {
                "label": "负面" if label_prob_tuple[0] == 0 else "非负面",
                "prob": float("{0:.5f}".format(label_prob_tuple[1])),
            }
            for label_prob_tuple in predict_results
        ]
        return handle_standard_response(StandardResponse(data=result))
    except Exception as e:
        request_data_log = json.dumps(
            {"title": title, "summary": summary, "client": client,},
            ensure_ascii=False,
            indent=2,
        )
        logger.error(f"/np [GET]接口错误。form data: {request_data_log}", exc_info=True)
        raise StandardResponse(msg=str(e), status_code=400, is_success=False)


def clean_text(text: str):
    return "".join(text.split())


@app.route("/np", methods=["POST"])
def np_batch():
    """
    计算问题相似度。
    :return:
    """
    try:
        request_json_dict = request.json
    except:
        if request.content_length < 1024:
            request_data_log = f"request body: {request.get_data(as_text=True)}"
        else:
            request_data_log = "request body 太大，不予输出"
        logger.error(f"/np [POST]接口错误。{request_data_log}", exc_info=True)
        raise StandardResponse(msg="不合法的json格式", status_code=400, is_success=False)
    try:
        samples = request_json_dict["samples"]

        # text_tuples = []
        text_list = []
        data = {"news":[],
                "titles":[],
                "labels":[]}
        for sample in samples:
            new = sample.get("new", "")
            title = sample.get("title", "")

            data["news"].append(new)
            data["titles"].append(title)
            data["labels"].append(0)

        data = TextPairDataset.from_dict_list(data)
        predict_result, loss = model.predict(data)
        result = [
            {"label": "负面" if t[0]==0 else "非负面",
             "probs": float("{0:.5f}".format(t[1]))}
            for t in predict_result
        ]

        with open("predict.jsonl", encoding="utf-8", mode="a+") as logfile:
            for sample_dict, predict_result_tupe in zip(samples, predict_result):
                sample_dict["label"] = predict_result_tupe[0]
                sample_dict["prob"] = predict_result_tupe[1].item()
                logfile.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")

        return handle_standard_response(StandardResponse(data=result))
    except Exception as e:
        if request.content_length < 1024:
            request_data_log = f"request body: {request.get_data(as_text=True)}"
        else:
            request_data_log = "request body 太大，不予输出"
        logger.error(f"/np [POST]接口错误。{request_data_log}", exc_info=True)
        raise StandardResponse(msg=str(e), status_code=400, is_success=False)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--m", dest="model_dir", help="模型存储目录", default="model", required=False
)

parser.add_argument(
    "-p",
    "--port",
    dest="port",
    help="web服务绑定的端口",
    default=7780,
    type=int,
    required=False,
)
parser.add_argument(
    "--host",
    dest="host",
    help="web服务绑定的host",
    default="0.0.0.0",
    type=str,
    required=False,
)

args = parser.parse_args()
model = BertSentSimCheckModel.load(args.model_dir)

from service_streamer import ThreadedStreamer

streamer = ThreadedStreamer(model.predict, batch_size=100, max_latency=0.01)

logger = logging.getLogger("model server")

logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

fh = logging.FileHandler(os.path.join("access.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logging.getLogger("werkzeug").addHandler(fh)
logging.getLogger("werkzeug").addHandler(default_handler)

app.config["JSON_AS_ASCII"] = False

app.run(port=args.port, host=args.host)
