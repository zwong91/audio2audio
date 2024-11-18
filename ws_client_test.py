import asyncio
import websockets
import base64

async def test_websocket():
    uri = "ws://108.136.246.72:8888/transcribe"  # WebSocket 服务器的地址
    
    # 读取音频文件并进行Base64编码
    audio_file_path = "tmp5tpgkw2o.wav"  # 替换成你自己的音频文件路径
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()  # 读取音频文件内容
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')  # 转换为Base64编码并解码为字符串

        print(f"Encoded audio data: {encoded_audio[:50]}...")  # 只打印前50个字符，避免输出过长

        # 连接到WebSocket服务器
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected")

            # 发送Base64编码的音频数据
            await websocket.send(encoded_audio)  # 发送消息
            print("Message sent")

            try:
                response = await websocket.recv()  # 接收消息
                print(f"Received message: {response}")

            except websockets.exceptions.ConnectionClosedOK as e:
                print(f"Connection closed normally: {e}")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")

    except Exception as e:
        print(f"Error connecting to WebSocket or reading audio file: {e}")

# 运行 WebSocket 客户端
asyncio.get_event_loop().run_until_complete(test_websocket())
