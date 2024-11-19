import asyncio
import websockets
import base64
import ssl

async def test_websocket():
    uri = "wss://108.136.246.72:5555/transcribe"  # WebSocket 服务器的地址
    
    # 读取音频文件并进行Base64编码
    audio_file_path = "tmp5tpgkw2o.wav"  # 替换成你自己的音频文件路径
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()  # 读取音频文件内容
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')  # 转换为Base64编码并解码为字符串

        print(f"Encoded audio data: {encoded_audio[:50]}...")  # 只打印前50个字符，避免输出过长

        # 创建 SSL 上下文，忽略证书验证
        ssl_context = ssl._create_unverified_context()

        # 连接到WebSocket服务器
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("WebSocket connected")

            async def send_ping():
                while True:
                    await asyncio.sleep(10)  # 每10秒发送一次ping
                    try:
                        await websocket.ping()
                    except Exception as e:
                        print(f"Error sending ping: {e}")
                        break

            ping_task = asyncio.create_task(send_ping())

            try:
                # 持续发送和接收消息
                while True:
                    # 发送Base64编码的音频数据
                    await websocket.send(encoded_audio)  # 发送消息
                    print("Message sent")

                    try:
                        response = await websocket.recv()  # 接收消息
                        print(f"Received message: {response}")

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