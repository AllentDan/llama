# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import json

from pathlib import Path
import triton_python_backend_utils as pb_utils

import asyncio
from urllib import response
import grpc

import inference_pb2
import inference_pb2_grpc


class Sender:

    def __init__(self, server_addresses) -> None:
        self.channels = [grpc.aio.insecure_channel(address) for address in server_addresses]
        self.clents = [inference_pb2_grpc.LLaMAServiceStub(channel) for channel in self.channels]

    async def send_request(self, message):
        # 创建消息对象
        request = inference_pb2.InferInput(prompts=message)

        # 发送请求到所有服务器
        tasks = [client.generate(request) for client in self.clents]
        responses = await asyncio.gather(*tasks)
        return responses[0].text

    async def close(self):
        await asyncio.gather(*[channel.close() for channel in self.channels])


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        # Instantiate the PyTorch model
        self.device = f"cuda:{args['model_instance_device_id']}"
        self.client = Sender(['localhost:50051', 'localhost:50052'])
        # self.channels = [grpc.aio.insecure_channel(address) for address in ['localhost:50051', 'localhost:50052']]
        # self.clents = [inference_pb2_grpc.LLaMAServiceStub(channel) for channel in self.channels]
        self.loop = asyncio.get_event_loop()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            import numpy as np
            prompt = pb_utils.get_input_tensor_by_name(request,
                                                       "INPUT0").as_numpy().astype(np.bytes_).tobytes().decode('utf-8')
            output = self.loop.run_until_complete(self.client.send_request(prompt))
            output = np.array([str(x).encode('utf-8') for x in output], dtype=np.bytes_)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", output.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
