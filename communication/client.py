
import grpc

import inference_pb2
from inference_pb2_grpc import LLaMAServiceStub


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = LLaMAServiceStub(channel)
        response = stub.generate(inference_pb2.InferInput(prompts=''))
        print(response)


if __name__ == '__main__':
    run()