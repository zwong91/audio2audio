import asyncio
import websockets
import base64
import ssl
import json
import time

async def test_websocket():
    uri = "wss://108.136.246.72:6666/transcribe"  # WebSocket 服务器的地址
    
    # 读取音频文件
    audio_file_path = "jp_zh.wav"  # 替换成你自己的音频文件路径
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()  # 读取音频文件内容

        # 创建 SSL 上下文，忽略证书验证
        ssl_context = ssl._create_unverified_context()
        # 记录开始时间
        start_time = time.time()
        # 连接到WebSocket服务器
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("WebSocket connected")
            # 记录结束时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"connected elapsed time: {elapsed_time:.2f} seconds")
            try:
                # 持续发送和接收消息
                frame_size = 320 * 2  # 320 个采样点，每个采样点 2 个字节
                for i in range(0, len(audio_data), frame_size):
                    audio_chunk = audio_data[i:i + frame_size]
                    encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')  # 转换为Base64编码并解码为字符串

                    # 创建要发送的 JSON 数据
                    data_to_send = [[[' 只是雨滴 受什么麻烦的这还没有打雷呢 ', '下雨总让人心情沉重呢。要不要聊聊？']], "Azure-xiaoxiao", encoded_audio]
                    json_data = json.dumps(data_to_send)

                    start_time = time.time()
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
                        await asyncio.sleep(0.1)  # 延迟0.1秒

                    except websockets.exceptions.ConnectionClosedOK as e:
                        print(f"Connection closed normally: {e}")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"Connection closed with error: {e}")
                        break

                # 发送静音数据
                silence_duration = 2  # 发送静音数据的持续时间（秒）
                silence_chunk = b'\x00' * frame_size
                for _ in range(int(silence_duration * 10)):  # 发送静音数据
                    encoded_audio = base64.b64encode(silence_chunk).decode('utf-8')
                    data_to_send = [[[' 只是雨滴 受什么麻烦的这还没有打雷呢 ', '下雨总让人心情沉重呢。要不要聊聊？']], "Azure-xiaoxiao", encoded_audio]
                    json_data = json.dumps(data_to_send)
                    await websocket.send(json_data)
                    print("Silence data sent")
                    await asyncio.sleep(0.1)  # 延迟0.1秒

            except Exception as e:
                print(f"Error during WebSocket communication: {e}")
    
    except Exception as e:
        print(f"Error connecting to WebSocket or reading audio file: {e}")

# 运行 WebSocket 客户端
asyncio.get_event_loop().run_until_complete(test_websocket())