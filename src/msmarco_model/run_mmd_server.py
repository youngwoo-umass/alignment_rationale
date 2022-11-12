import os
from cpath import get_mmd_model_path, MMD_PORT
from tf_v2_support import disable_eager_execution
from msmarco_model.bert_like_server import RPCServerWrap
from msmarco_model.mmd_server import PredictorClsDense


def run_server():
    save_path = get_mmd_model_path()

    disable_eager_execution()

    predictor = PredictorClsDense(2, 512)
    predictor.load_model(save_path)

    def predict(payload):
        sout = predictor.predict(payload)
        return sout

    server = RPCServerWrap(predict)
    print("server started")
    server.start(MMD_PORT)


if __name__ == "__main__":
    run_server()