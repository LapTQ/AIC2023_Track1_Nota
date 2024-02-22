from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent

sys.path.append(str(ROOT_DIR))

from protos.python.feature_extraction import feature_extraction_pb2_grpc, feature_extraction_pb2
from trackers.botsort.fast_reid_interfece import FastReIDInterface

# =============================================================================

import grpc
import pickle
from concurrent import futures
import argparse
import socket
from pytorch_lightning.trainer.trainer import Trainer
import torch
import numpy as np
import cv2


class FeatureExtractionServicer(feature_extraction_pb2_grpc.FeatureExtractorServicer):

    def __init__(
            self, 
            **kwargs
    ):
        config_file = kwargs["config_file"]
        weights_path = kwargs["weights_path"]
        self.device = kwargs["device"]
        batch_size = kwargs["batch_size"]

        self.model = FastReIDInterface(
            config_file=config_file, 
            weights_path=weights_path,
            device=self.device,
            batch_size=batch_size
        )


    def predict(self, request, context):
        img_batch = pickle.loads(request.imgs_pkl)

        with torch.no_grad():
            features = []
            print(f'[{np.random.randint(1, 10)} :-D] Processing batch of size', img_batch.size(0))
            for img in img_batch:
                img = img.numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                g = 2.0
                img = img.astype(np.float64)
                img = ((img / 255) ** (1 / g)) * 255
                img = img.astype(np.uint8)
                dets = np.array([[0, 0, img.shape[1] - 1, img.shape[0] - 1]])
                feature = self.model.inference(img, dets)[0]
                # feature = feature / np.linalg.norm(feature)
                features.append(feature)
            features = torch.tensor(features)

        features_pkl = pickle.dumps(features)
        return feature_extraction_pb2.FeatureBatch(features_pkl=features_pkl)


def parse_kwargs():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--port", required=True, type=int, help="Port number")
    ap.add_argument("--device", required=True, type=str)
    ap.add_argument("--config_file", required=True, type=str)
    ap.add_argument("--weights_path", required=True, type=str)
    ap.add_argument("--batch_size", required=True, type=int)

    kwargs = vars(ap.parse_args())

    return kwargs


if __name__ == '__main__':

    kwargs = parse_kwargs()
    
    port = kwargs["port"]

    # check if the port is not used
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            raise ValueError(f"Port {port} is already in use. Please check if the server is already running.")
        
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    feature_extraction_pb2_grpc.add_FeatureExtractorServicer_to_server(FeatureExtractionServicer(**kwargs), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started at port {port}")
    server.wait_for_termination()