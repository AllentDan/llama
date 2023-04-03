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
        input0_aligned = np.zeros((1, 2048), dtype=np.bytes_)
        input0_aligned[0, :input0_data.shape[0]] = input0_data
        input_len = np.array([input0_data.shape[0]], dtype=np.int32).reshape(1, -1)
        inputs = [
            httpclient.InferInput("INPUT0", input0_aligned.shape, "BYTES"),
            httpclient.InferInput("INPUT_LEN", input_len.shape, "INT32"),
        ]

        inputs[0].set_data_from_numpy(input0_aligned)
        inputs[1].set_data_from_numpy(input_len)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
            httpclient.InferRequestedOutput("OUTPUT_LEN"),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0").astype(np.bytes_)
        output_len = response.as_numpy("OUTPUT_LEN").astype(np.int32)
        for i in range(int(output0_data.shape[0])):
            output = output0_data[i, :int(output_len[i])].tobytes().decode('utf-8', 'ignore')
            print(output)
            # print(output0_data.tobytes().decode('utf-8'))


if __name__ == '__main__':
    args = parse_args()
    output = main(args.model_name, args.query, args.url)
