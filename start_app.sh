#!/bin/bash

# Start the app
python3 -m src.main --certfile cf.pem --keyfile cf.key --port 5555 --vad-type pyannote --vad-args '{"auth_token": "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"}'