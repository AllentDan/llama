# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import triton_python_backend_utils as pb_utils
import asyncio
import grpc
import threading
import time
import numpy as np

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
        output_len_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_LEN")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.output_len_dtype = pb_utils.triton_string_to_numpy(output_len_config['data_type'])

        # Instantiate the PyTorch model
        self.device = f"cuda:{args['model_instance_device_id']}"
        self.client = Sender(['localhost:50051', 'localhost:50052'])
        # self.channels = [grpc.aio.insecure_channel(address) for address in ['localhost:50051', 'localhost:50052']]
        # self.clents = [inference_pb2_grpc.LLaMAServiceStub(channel) for channel in self.channels]
        self.loop = asyncio.get_event_loop()
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. The request.get_response_sender() must be used to
        get an InferenceResponseSender object associated with the request.
        Use the InferenceResponseSender.send(response=<infer response object>,
        flags=<flags>) to send responses.

        In the final response sent using the response sender object, you must
        set the flags argument to TRITONSERVER_RESPONSE_COMPLETE_FINAL to
        indicate no responses will be sent for the corresponding request. If
        there is an error, you can set the error argument when creating a
        pb_utils.InferenceResponse. Setting the flags argument is optional and
        defaults to zero. When the flags argument is set to
        TRITONSERVER_RESPONSE_COMPLETE_FINAL providing the response argument is
        optional.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        None
        """

        # Visit individual request to start processing them. Note that execute
        # function is not required to wait for all the requests of the current
        # batch to be processed before returning.
        for request in requests:
            self.process_request(request)

        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another batch of
        # requests. As we are not waiting for the response thread to complete
        # here, it is possible that at any give time the model may be processing
        # multiple batches of requests. Depending upon the request workload,
        # this may lead to a lot of requests being processed by a single model
        # instance at a time. In real-world models, the developer should be
        # mindful of when to return from execute and be willing to accept next
        # request batch.
        return None

    def process_request(self, request):
        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        # Get INPUT0
        prompt_len = pb_utils.get_input_tensor_by_name(request, "INPUT_LEN").as_numpy().astype(np.int32)
        input_array = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy().astype(np.bytes_)
        # print(prompt_len.shape, input_array.shape)
        prompts = []
        for i in range(input_array.shape[0]):
            prompt = input_array[i, :int(prompt_len[i])].tobytes().decode('utf-8', 'ignore')
            prompts.append(prompt)

        thread = threading.Thread(target=self.response_thread, args=(request.get_response_sender(), prompts))

        # A model using decoupled transaction policy is not required to send all
        # responses for the current request before returning from the execute.
        # To demonstrate the flexibility of the decoupled API, we are running
        # response thread entirely independent of the execute thread.
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, prompts):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.

        for idx in range(1):
            outputs = self.loop.run_until_complete(self.client.send_request(prompts))
            outputs_aligned = np.zeros((len(outputs), 2048), dtype=np.bytes_)
            outputs_len = np.zeros((len(outputs), 1), dtype=np.int32)
            for i, output in enumerate(outputs):
                output = np.array([str(x).encode('utf-8') for x in output], dtype=np.bytes_)
                outputs_aligned[i, :int(output.shape[0])] = output
                outputs_len[i, 0] = int(output.shape[0])

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", outputs_aligned.astype(self.output0_dtype))
            out_tensor_len = pb_utils.Tensor("OUTPUT_LEN", outputs_len.astype(self.output_len_dtype))

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_len])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Finalize invoked')

        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = (logging_time_sec / sleep_time_sec)
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = (self.inflight_thread_count != 0)
                if (cycles % cycle_to_log == 0):
                    print(f"Waiting for {self.inflight_thread_count} response threads to complete...")
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1

        print('Finalize complete...')
