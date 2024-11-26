module.exports = {
  apps: [
    {
      name: "rt-audio",
      script: "/home/ubuntu/miniconda3/envs/chattts/bin/uvicorn",
      args: "app_ws:app --host 0.0.0.0 --port 5555 --ssl-keyfile cf.key --ssl-certfile cf.pem --log-level debug",
      cwd: "/workspaces/audio2audio/voice_chat",
      interpreter: "/home/ubuntu/miniconda3/envs/chattts/bin/python",
      env: {
        CONDA_DEFAULT_ENV: "chattts",
        OPENAI_API_KEY: "sk-xxxxx",
      },
    },
  ],
};