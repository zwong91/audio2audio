module.exports = {
    apps: [
      {
        name: "rt-audio2",
        script: "python3",
        args: "-m src.main --certfile cf.pem --keyfile cf.key",
        cwd: "/home/ubuntu/proj/rt-audio", // 替换为你的项目目录
        interpreter: "/home/ubuntu/miniconda3/envs/rt/bin/python", // 替换为你的 conda 环境中的 Python 解释器路径
        env: {
          CONDA_DEFAULT_ENV: "rt", // 替换为你的 conda 环境名称
          OPENAI_API_KEY: "sk-xxxx"  // 替换为你的 OpenAI API Key
        },
      },
    ],
  };