import asyncio
import websockets
import base64
import ssl
import json
import time

async def test_websocket():
    uri = "wss://108.136.246.72:6666/transcribe"  # WebSocket 服务器的地址
    
    # 读取音频文件并进行Base64编码
    audio_file_path = "jp_zh.wav"  # 替换成你自己的音频文件路径
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()  # 读取音频文件内容
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')  # 转换为Base64编码并解码为字符串

        print(f"Encoded audio data: {encoded_audio[:50]}...")  # 只打印前50个字符，避免输出过长
        # 创建要发送的 JSON 数据
        data_to_send = [[[' 只是雨滴 受什么麻烦的这还没有打雷呢 ', '下雨总让人心情沉重呢。要不要聊聊？']], "Azure-xiaoxiao", encoded_audio]
        json_data = json.dumps(data_to_send)

        # 创建 SSL 上下文，忽略证书验证
        ssl_context = ssl._create_unverified_context()
        # 记录开始时间
        start_time = time.time()
        # 连接到WebSocket服务器
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("WebSocket connected")

            try:
                # 持续发送和接收消息
                while True:
                    # 发送Base64编码的音频数据
                    await websocket.send(json_data)  # 发送消息
                    print("JSON data sent")

                    try:
                        response = await websocket.recv()  # 接收消息
                        #print(f"Received message: {response}")
                        # 解码接收到的消息
                        decoded_response = json.loads(response)
                        print(f"Decoded message: {decoded_response}")
                        # 记录结束时间
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Total elapsed time: {elapsed_time:.2f} seconds")

                        # 添加延迟
                        await asyncio.sleep(2)  # 延迟2秒

                    except websockets.exceptions.ConnectionClosedOK as e:
                        print(f"Connection closed normally: {e}")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"Connection closed with error: {e}")
                        break

            finally:
                ping_task.cancel()
                await ping_task

    except Exception as e:
        print(f"Error connecting to WebSocket or reading audio file: {e}")

# 运行 WebSocket 客户端
asyncio.get_event_loop().run_until_complete(test_websocket())