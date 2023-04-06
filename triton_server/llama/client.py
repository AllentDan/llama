# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import queue
from functools import partial


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class gRPCClient:

    def __init__(self, model_name, server_addr):
        self.client = grpcclient.InferenceServerClient(server_addr)
        self.user_data = UserData()
        self.model_name = model_name

    def start_stream(self, callback: callable):
        """ Starts a grpc bi-directional stream to send streaming inferences.
            Note: When using stream, user must ensure the gRPCClient.close()
            gets called at exit.
        
        Examples:
            >>> from functools import partial
            >>> import gPRCClient
            >>>
            >>>
            >>> gRPCClient client("localhost:8000")
            >>> def callback(client, req_id, text):
            >>>     if len(text):
            >>>         print(text)
            >>>     else:
            >>>         client.close_stream()
            >>>         
            >>> 
            >>> client.start_stream(callback=partial(callback, client))
            >>> client.async_stream_infer("0x1a2b3c4d5f", "how to learn pytorch?")
            
        Args:
            callback(function): Python function that is invoked upon receiving response from
                the underlying stream. The function must reserve the last two
                arguments (result, error) to hold InferResult and
                InferenceServerException objects respectively which will be
                provided to the function when executing the callback. The
                ownership of these objects will be given to the user. The
                'error' would be None for a successful inference.
        """
        self.client.start_stream(callback=partial(callback, self.user_data))

    def async_stream_infer(self, req_id: str, prompt: str):
        """ request the server to generate an answer to the prompt. This is an 
        
        Args
            req_id(str): an identical id of the request
            prompt(str): the requested prompt
        """
        input0_data = np.array([str(x).encode('utf-8') for x in prompt], dtype=np.bytes_)
        input0_aligned = np.zeros((1, 2048), dtype=np.bytes_)
        input0_aligned[0, :input0_data.shape[0]] = input0_data
        input_len = np.array([input0_data.shape[0]], dtype=np.int32).reshape(1, -1)
        inputs = [
            grpcclient.InferInput("INPUT0", input0_aligned.shape, "BYTES"),
            grpcclient.InferInput("INPUT_LEN", input_len.shape, "INT32"),
        ]

        inputs[0].set_data_from_numpy(input0_aligned)
        inputs[1].set_data_from_numpy(input_len)

        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
            grpcclient.InferRequestedOutput("OUTPUT_LEN"),
        ]
        self.client.async_stream_infer(model_name=self.model_name, inputs=inputs, request_id=req_id, outputs=outputs)
        while True:
            data_item = self.user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item
            output0_data = data_item.as_numpy("OUTPUT0").astype(np.bytes_)
            output_len = data_item.as_numpy("OUTPUT_LEN").astype(np.int32)
            # 没能在客户端获取到 TRITONSERVER_RESPONSE_COMPLETE_FINAL 标志，自定义一个空的输出，如果客户端识别就当结束传输了
            if output_len.sum() == 0:
                break
            for i in range(int(output0_data.shape[0])):
                output = output0_data[i, :int(output_len[i])].tobytes().decode('utf-8', 'ignore')
                print(output)

    def close_stream(self):
        """ Close the stream created by `start_stream`
        """
        self.client.stop_stream()


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('query', type=str)
    parser.add_argument('--model_name', type=str, required=False, default="llama")
    parser.add_argument('-u', '--url', type=str, required=False, default='0.0.0.0:8010')
    return parser.parse_args()


def main(model_name: str, prompt: str, url: str):
    client = gRPCClient(model_name, url)
    client.start_stream(callback=callback)
    client.async_stream_infer('0', prompt)


if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, args.query, args.url)
