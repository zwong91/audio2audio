// page.tsx
"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

export default function VoiceCall() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<string>("连接中...");
  const [callDuration, setCallDuration] = useState<number>(0);

  let audioContext: AudioContext | null = null;
  let audioBufferQueue: AudioBuffer[] = [];

  if (typeof window !== "undefined" && window.AudioContext) {
    audioContext = new AudioContext();
  }

  // 添加通话计时器
  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const audioManager = {
    stopCurrentAudio: () => {
      if (isPlayingAudio) {
        setIsPlayingAudio(false);
      }
    },

    playNewAudio: async (audioBlob: Blob) => {
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onloadedmetadata = () => {
        setAudioDuration(audio.duration);
      };

      setIsPlayingAudio(true);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsPlayingAudio(false);
        setIsRecording(true);

        setTimeout(() => {
          if (audioQueue.length > 0) {
            const nextAudioBlob = audioQueue.shift();
            if (nextAudioBlob) {
              audioManager.playNewAudio(nextAudioBlob);
            }
          }
        }, 100);
      };

      try {
        await audio.play();
      } catch (error) {
        console.error("播放失败:", error);
        audioManager.stopCurrentAudio();
      }
    }
  };

  function bufferAudio(data: ArrayBuffer) {
    if (audioContext) {
      audioContext.decodeAudioData(data, (buffer) => {
        audioBufferQueue.push(buffer);
        if (!isPlayingAudio) {
          playAudioBufferQueue();
        }
      });
    }
  }

  function playAudioBufferQueue() {
    if (audioBufferQueue.length === 0) {
      setIsPlayingAudio(false);
      setIsRecording(true);
      return;
    }

    const buffer = audioBufferQueue.shift();
    if (buffer && audioContext) {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.onended = () => {
        playAudioBufferQueue();
      };
      source.start();
      setIsPlayingAudio(true);
    }
  }

  // WebSocket连接和音频处理逻辑
  useEffect(() => {
    const SOCKET_URL = "wss://gtp.aleopool.cc/stream";
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
    
    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;

      navigator.mediaDevices?.getUserMedia({ audio: true })
        .then((stream) => {
          let websocket: WebSocket | null = null;

          const reconnectWebSocket = () => {
            if (websocket) websocket.close();
            websocket = new WebSocket(SOCKET_URL);
            setSocket(websocket);

            websocket.onopen = () => {
              setConnectionStatus("已连接");
              const recorder = new RecordRTC(stream, {
                type: 'audio',
                recorderType: StereoAudioRecorder,
                mimeType: 'audio/wav',
                timeSlice: 500,
                desiredSampRate: 16000,
                numberOfAudioChannels: 1,
                ondataavailable: handleAudioData
              });

              recorder.startRecording();
            };

            websocket.onmessage = handleWebSocketMessage;
            websocket.onclose = handleWebSocketClose;
            websocket.onerror = handleWebSocketError;
          };

          reconnectWebSocket();
        })
        .catch(error => console.error("麦克风访问失败:", error));
    };

    document.body.appendChild(script);

    return () => {
      socket?.close();
    };
  }, []);

  // 音频数据处理函数
  const handleAudioData = (blob: Blob) => {
    if (blob.size > 0) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result) {
          const base64data = arrayBufferToBase64(reader.result as ArrayBuffer);
          const dataToSend = [[], "xiaoxiao", base64data];
          socket?.send(JSON.stringify(dataToSend));
        }
      };
      reader.readAsArrayBuffer(blob);
    }
  };

  // WebSocket消息处理函数
  const handleWebSocketMessage = (event: MessageEvent) => {
    setIsRecording(false);
    setIsPlayingAudio(true);

    try {
      let audioData: ArrayBuffer;
      if (event.data instanceof ArrayBuffer) {
        audioData = event.data;
      } else if (event.data instanceof Blob) {
        const reader = new FileReader();
        reader.onloadend = () => {
          audioData = reader.result as ArrayBuffer;
          bufferAudio(audioData);
        };
        reader.readAsArrayBuffer(event.data);
        return;
      } else {
        throw new Error("未知的数据类型");
      }
      bufferAudio(audioData);
    } catch (error) {
      console.error("音频处理失败:", error);
    }
  };

  const handleWebSocketClose = () => {
    console.log("WebSocket连接已断开，正在重连...");
    setConnectionStatus("重新连接中...");
    setTimeout(reconnectWebSocket, 5000);
  };

  const handleWebSocketError = (error: Event) => {
    console.error("WebSocket错误:", error);
    socket?.close();
  };

  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    const uint8Array = new Uint8Array(arrayBuffer);
    let binary = '';
    uint8Array.forEach(byte => binary += String.fromCharCode(byte));
    return btoa(binary);
  }

  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div className={`${styles.statusDot} ${connectionStatus === '已连接' ? styles.connected : ''}`} />
          {connectionStatus}
        </div>
        <div className={styles.duration}>{formatTime(callDuration)}</div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.avatarSection}>
          <div className={`${styles.avatarContainer} ${isPlayingAudio ? styles.speaking : ''}`}>
            <img src="/ai-avatar.png" alt="AI" className={styles.avatar} />
            <div className={styles.audioWaves}>
              {Array(3).fill(0).map((_, i) => (
                <div key={i} className={`${styles.wave} ${isPlayingAudio ? styles.active : ''}`} />
              ))}
            </div>
          </div>
          <div className={styles.status}>
            {isPlayingAudio ? "AI正在说话" : "AI正在听"}
          </div>
        </div>
      </div>

      <div className={styles.controls}>
        <button 
          className={styles.endCallButton}
          onClick={() => window.location.reload()}
        >
          结束通话
        </button>
      </div>
    </div>
  );
}