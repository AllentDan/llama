# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from tritonclient.utils import *
import tritonclient.http as httpclient


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('query', type=str)
    parser.add_argument('--model_name', type=str, required=False, default="llama")
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8010')
    return parser.parse_args()


def main(model_name: str, prompt: str, url: str):

    with httpclient.InferenceServerClient(url) as client:
        input0_data = np.array([str(x).encode('utf-8') for x in prompt], dtype=np.bytes_)
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape, "BYTES"),
        ]
        # print(input0_data)

        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")
        print(output0_data.astype(np.bytes_).tobytes().decode('utf-8'))


if __name__ == '__main__':
    args = parse_args()
    output = main(args.model_name, args.query, args.url)
