
import asyncio
from urllib import response
import grpc

import inference_pb2
import inference_pb2_grpc


class Sender:
    def __init__(self, server_addresses) -> None:
        self.channels = [grpc.aio.insecure_channel(address)
                         for address in server_addresses]
        self.clents = [inference_pb2_grpc.LLaMAServiceStub(channel)
                       for channel in self.channels]

    async def send_request(self, message):
        # 创建消息对象
        request = inference_pb2.InferInput(prompts=message)

        # 发送请求到所有服务器
        tasks = [client.generate(request) for client in self.clents]
        responses = await asyncio.gather(*tasks)
        print(f"{responses[0].text}")

    async def close(self):
        await asyncio.gather(*[channel.close() for channel in self.channels])


async def send_request(server_address, message):
    # 创建gRPC通道和异步客户端
    channel = grpc.aio.insecure_channel(server_address)
    client = inference_pb2_grpc.LLaMAServiceStub(channel)

    # 创建异步版本的消息对象
    request = inference_pb2.InferInput(prompts=message)

    # 发送异步请求
    response = await client.generate(request)
    if server_address == 'localhost:50051':
        # 处理响应
        print(f"Server {server_address}: {response.text}")


async def main():
    client = Sender(['localhost:50051', 'localhost:50052'])
    print('please input prompt:')
    while True:
        print('>>')
        prompt = input()
        if prompt != 'exit':
            await client.send_request(prompt)
        else:
            await client.close()

    # task1 = asyncio.create_task(send_request('localhost:50051', 'Hello from client 1'))
    # task2 = asyncio.create_task(send_request('localhost:50052', 'Hello from client 2'))

    # # 等待所有任务完成
    # await asyncio.gather(task1, task2)

if __name__ == '__main__':
    asyncio.run(main())
