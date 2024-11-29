import asyncio
import json
import logging
import ssl
import uuid
import base64
import uvicorn
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from src.client import Client

class Server:
    """
    WebSocket server for real-time audio transcription with VAD and ASR pipelines.
    """

    def __init__(
        self,
        vad_pipeline,
        asr_pipeline,
        llm_pipeline,
        tts_pipeline,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
        static_dir="assets",  # 静态文件目录
    ):
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.llm_pipeline = llm_pipeline
        self.tts_pipeline = tts_pipeline
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.connected_clients = {}
        
        # Initialize FastAPI app
        self.app = FastAPI()
        # 配置 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Add HTTP routes (like rendering HTML)
        self.templates = Jinja2Templates(directory="templates")

        # Add route to serve static files from the 'assets' directory
        self.app.add_event_handler("startup", self.startup)
        #self.app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

        # Define additional HTTP routes
        self.app.get("/asset/{filename}")(self.get_asset_file)
           
        # Add WebSocket route for audio transcription
        self.app.websocket("/transcribe")(self.websocket_endpoint)

    async def startup(self):
        """Called on startup to set up additional services."""
        logging.info(f"Starting server at {self.host}:{self.port}")

    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client
        logging.info(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        finally:
            del self.connected_clients[client_id]
            logging.info(f"Client {client_id} disconnected")

    async def handle_audio(self, client, websocket):
        while True:
            try:
                payload = await websocket.receive_text()
                message = json.loads(payload)
                bytes = base64.b64decode(message[2])
                client.append_audio_data(bytes, message[0], message[1])
                if isinstance(message, str):
                    await self.handle_text_message(client, message)
                else:
                    logging.warning(f"Unexpected message type from {client.client_id}")
                
                client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline, self.llm_pipeline, self.tts_pipeline)
            except WebSocketDisconnect as e:
                logging.error(f"Connection with {client.client_id} closed: {e}")
                break
            except Exception as e:
                logging.error(f"Error handling audio for {client.client_id}: {e}")
                break

    async def handle_text_message(self, client, message):
        """Handles incoming JSON text messages for config updates."""
        try:
            config = json.loads(message)
            if config.get("type") == "config":
                client.update_config(config["data"])
                logging.debug(f"Updated config: {client.config}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode config message: {e}")

    async def get_asset_file(self, filename: str):
        file_path = os.path.join('/tmp', filename)
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.webm': 'audio/webm'
        }
        ext = os.path.splitext(filename)[1].lower()
        media_type = mime_types.get(ext, 'application/octet-stream')

        return FileResponse(
            path=file_path,
            media_type=media_type,
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Disposition': 'inline'
            }
        )
    
    def create_uvicorn_server(self, ssl_context=None):
        """Creates and returns a Uvicorn server instance."""
        uvicorn_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            ssl_certfile=self.certfile,
            ssl_keyfile=self.keyfile,
            loop="uvloop",
            log_level="debug",
            workers=os.cpu_count(),
            limit_concurrency=1000,
            limit_max_requests=10000,
            backlog=2048
        )
        server = uvicorn.Server(uvicorn_config)
        return server

    def start(self):
        """Start the WebSocket server."""
        ssl_context = None
        if self.certfile and self.keyfile:
            logging.info(f"Starting secure WebSocket server on {self.host}:{self.port}")
        else:
            logging.info(f"Starting WebSocket server on {self.host}:{self.port}")

        server = self.create_uvicorn_server(ssl_context)
        server.run()

