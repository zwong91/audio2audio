import asyncio
import websockets

async def test_websocket():
    uri = "wss://audio.xyz666.org:8443/transcribe"
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected")

            await websocket.send("Hello, server!")  # 发送消息
            print("Message sent")

            try:
                response = await websocket.recv()  # 接收消息
                print(f"Received message: {response}")

            except websockets.exceptions.ConnectionClosedOK as e:
                print(f"Connection closed normally: {e}")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")

    except Exception as e:
        print(f"Error connecting to WebSocket: {e}")

# 运行 WebSocket 客户端
asyncio.get_event_loop().run_until_complete(test_websocket())
