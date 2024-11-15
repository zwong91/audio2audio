import time
from gradio_client import Client, file

# 记录开始时间
start_time = time.time()

client = Client("https://ea7f3c90fff14c7c8a.gradio.live")
result = client.predict(
    audio=file('https://funaudiollm.github.io/audios/s2st/zh/zh_prompt.wav'),
    history=[],
    api_name="/model_chat"
)

# 记录结束时间
end_time = time.time()

# 计算总耗时
elapsed_time = end_time - start_time

print("预测结果:", result)
print(f"耗时: {elapsed_time:.2f} 秒")