
import asyncio
from urllib import response
import grpc

import inference_pb2
import inference_pb2_grpc


class Sender:
    def __init__(self, server_addresses) -> None:
        self.channels = [grpc.aio.insecure_channel(address)
                         for address in server_addresses]
        self.clients = [inference_pb2_grpc.LLaMAServiceStub(channel)
                       for channel in self.channels]

    async def async_request(self, prompts):
        # 创建消息对象
        request = inference_pb2.InferInput(prompts=[prompts])

        # 发送请求到所有服务器
        tasks = [client.generate(request) for client in self.clents]
        responses = await asyncio.gather(*tasks)
        for reponse in responses:
            if response.rank == 0:
                print(f"{response.text}")

        
    async def stream_request(self, prompts):
        request = inference_pb2.InferInput(prompts=[prompts])

        async def read_responses(call):
            async for response in call:
                if response.rank == 0:
                    for text in response.texts:
                        print(text)

        await asyncio.gather(
            *(read_responses(client.GenerateServerStream(request))
              for client in self.clients))
        
    async def close(self):
        await asyncio.gather(*[channel.close() for channel in self.channels])


async def main():
    client = Sender(['localhost:50051', 'localhost:50052'])
    print('please input prompt:')
    while True:
        print('>>')
        prompt = input()
        if prompt != 'exit':
            await client.stream_request(prompt)
        else:
            await client.close()


if __name__ == '__main__':
    asyncio.run(main())
