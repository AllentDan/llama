from base64 import decode
from concurrent import futures

import asyncio
from multiprocessing import dummy
from urllib import response
import grpc
import inference_pb2
import inference_pb2_grpc

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import threading
import queue


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} "
    "but world size is {world_size}"

    ckpt_path = checkpoints[local_rank]
    print(f"Loading {local_rank}, {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


class LLaMAServer(inference_pb2_grpc.LLaMAServiceServicer):
    def __init__(self, local_rank: int = 0, generator: LLaMA = None) -> None:
        super().__init__()
        self.count = 0
        self.local_rank = local_rank
        self.generator = generator

    async def decode(self, tokens, prompt_tokens, pre_pos, cur_pos):
        str = 'dummy decode: {}, {}'.format(pre_pos, cur_pos)
        return str
    
    async def dummy(self, num):
        for i in range(num):
            tokens = None
            prompt_tokens = None
            pre_pos = 0
            cur_pos = i
            yield tokens, prompt_tokens, pre_pos, cur_pos
            await asyncio.sleep(.1)
            
    # async def Generate(self, request_iterator, context):
    #     # async for request in request_iterator:
    #     #     print(f'[{self.local_rank}], prompt: {request.prompts}')
    #     async for tokens, prompt_tokens, pre_pos, cur_pos in self.dummpy(10):
    #         text = await self.decode(tokens, prompt_tokens, pre_pos, cur_pos)
    #         print(text)
    #         response = inference_pb2.InferOutput(text=text)
    #         yield response
            
    async def Generate(self, request, conext):
        print(f'[{self.local_rank}], prompt: {request.prompts}')

        message = self.generator.generate(
            [request.prompts],
            max_gen_len=256,
            temperature=0.8,
            top_p=0.95)
        # message = self.dummy(10)
        # async for tokens, prompt_tokens, pre_pos, cur_pos in message:
        #     text = await self.decode(tokens, prompt_tokens, pre_pos, cur_pos)
        #     print(text)
        #     response = inference_pb2.InferOutput(text=text)
        #     yield response
        pre_pos = 0
        async for tokens, prompt_tokens, cur_pos in message:
            # print(f'cur: {cur_pos}')
            if cur_pos - pre_pos >= 50:
                text = await self.generator.decode(tokens, prompt_tokens, 
                                             pre_pos, cur_pos)
                print(f'cur: {cur_pos}, {text}')
                pre_pos = cur_pos
                response = inference_pb2.InferOutput(text=text[0])
                yield response
        # if cur_pos > pre_pos:
        #     text = await self.generator.decode(tokens, prompt_tokens, 
        #                                  pre_pos, cur_pos)
        #     print(text)
        #     response = inference_pb2.InferOutput(text=text)
        #     yield response
    #     # # print(f'[{self.local_rank}], message: {text}')
    #     # # return inference_pb2.InferOutput(text='')
    #     # # return inference_pb2.InferOutput(text=message[0])
       

    def commit(self, request, context):
        print('this is server: commit')
        pass

    def get(self, request, conext):
        print('this is server: get')
        pass


async def async_server(local_rank, port, generator):
    server = grpc.aio.server()
    inference_pb2_grpc.add_LLaMAServiceServicer_to_server(
        LLaMAServer(local_rank=local_rank, generator=generator),
        server)
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    print(f'async gRPC starting on {listen_addr}...')
    await server.start()
    await server.wait_for_termination()


def server(port, generator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_LLaMAServiceServicer_to_server(
        LLaMAServer(generator),
        server)
    server.add_insecure_port(f'[::]:{port}')
    print('gRPC starting...')
    server.start()
    server.wait_for_termination()


def worker(generator):
    result = generator.generate([''], max_gen_len=256, temperature=0.8, top_p=0.95)
    print(result)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    port: int = 50051
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    port += local_rank

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len,
        max_batch_size
    )

    asyncio.run(async_server(local_rank, port, generator))


if __name__ == '__main__':
    fire.Fire(main)
