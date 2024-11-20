module.exports = {
    apps: [
      {
        name: "rt-audio",
        script: "app_ws.py",
        cwd: "/workspaces/audio2audio/voice_chat", // 替换为你的项目目录
        interpreter: "/home/ubuntu/miniconda3/envs/chattts/bin/python", // 替换为你的 conda 环境中的 Python 解释器路径
        env: {
          CONDA_DEFAULT_ENV: "chattts", // 替换为你的 conda 环境名称
          OPENAI_API_KEY: ""  // 替换为你的 OpenAI API Key
        },
      },
    ],
  };