"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import styles from "./page.module.css";

export default function Home() {
  // 状态定义
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);
  const [currentAudioElement, setCurrentAudioElement] = useState<HTMLAudioElement | null>(null);

  // 类型定义
  type HistoryItem = [string, string];
  type History = HistoryItem[];
  const [history, setHistory] = useState<History>([]);
  
  const SOCKET_URL = "wss://gtp.aleopool.cc/stream";

  // 音频控制函数
  const stopCurrentAudio = useCallback(() => {
    if (currentAudioElement) {
      currentAudioElement.pause();
      currentAudioElement.currentTime = 0;
      URL.revokeObjectURL(currentAudioElement.src);
      setCurrentAudioElement(null);
      setIsPlayingAudio(false);
    }
  }, [currentAudioElement]);

  const playNewAudio = useCallback(async (audioBlob: Blob) => {
    stopCurrentAudio();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    
    setCurrentAudioElement(audio);
    setIsPlayingAudio(true);
    
    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
      setCurrentAudioElement(null);
      setIsPlayingAudio(false);
      setIsRecording(true);
      processQueue();
    };

    await audio.play();
  }, [stopCurrentAudio]);

  const processQueue = useCallback(async () => {
    if (audioQueue.length === 0) {
      setIsProcessingQueue(false);
      return;
    }

    setIsProcessingQueue(true);
    const nextBlob = audioQueue[0];
    setAudioQueue(prevQueue => prevQueue.slice(1));

    try {
      await playNewAudio(nextBlob);
    } catch (error) {
      console.error("播放音频失败:", error);
      setIsProcessingQueue(false);
    }
  }, [audioQueue, playNewAudio]);

  const audioManager = useMemo(() => ({
    stopCurrentAudio,
    playNewAudio,
    processQueue,
    addToQueue: (audioBlob: Blob) => {
      // 直接替换队列内容，保持最大长度为1
      setAudioQueue([audioBlob]);
      if (!isProcessingQueue) {
        processQueue();
      }
    }
  }), [stopCurrentAudio, playNewAudio, processQueue, isProcessingQueue]);

  // 工具函数
  const arrayBufferToBase64 = (arrayBuffer: ArrayBuffer): string => {
    const uint8Array = new Uint8Array(arrayBuffer);
    return btoa(String.fromCharCode.apply(null, Array.from(uint8Array)));
  };

  // 屏幕常亮
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;

    async function requestWakeLock() {
      try {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired");
      } catch (error) {
        console.error("Failed to acquire wake lock", error);
      }
    }

    requestWakeLock();

    return () => {
      wakeLock?.release().then(() => {
        console.log("Screen wake lock released");
      }).catch(console.error);
    };
  }, []);

  // 音频设备初始化
  useEffect(() => {
    const initMediaDevices = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        console.error("Media devices API not supported.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setMediaRecorder(new MediaRecorder(stream));
      } catch (error) {
        console.error("Error accessing media devices:", error);
      }
    };

    initMediaDevices();
  }, []);

  // WebRTC 初始化
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
    
    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;

      const initWebSocket = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          let websocket: WebSocket | null = null;

          const reconnectWebSocket = () => {
            if (websocket) websocket.close();
            websocket = new WebSocket(SOCKET_URL);
            setSocket(websocket);

            websocket.onopen = () => {
              console.log("Connected to websocket");
              const recorder = new RecordRTC(stream, {
                type: 'audio',
                recorderType: StereoAudioRecorder,
                mimeType: 'audio/wav',
                timeSlice: 500,
                desiredSampRate: 16000,
                numberOfAudioChannels: 1,
                ondataavailable: (blob: Blob) => {
                  if (blob.size > 0) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                      if (reader.result) {
                        const base64data = arrayBufferToBase64(reader.result as ArrayBuffer);
                        const dataToSend = [history, "xiaoxiao", base64data];
                        websocket?.send(JSON.stringify(dataToSend));
                      }
                    };
                    reader.readAsArrayBuffer(blob);
                  }
                }
              });

              recorder.startRecording();
            };

            websocket.onmessage = (event) => {
              setIsRecording(false);
              setIsPlayingAudio(true);

              try {
                const jsonData = JSON.parse(event.data);
                const { stream: audioBase64, history: receivedHistory } = jsonData;

                if (Array.isArray(receivedHistory)) {
                  const formattedHistory: History = receivedHistory
                    .filter((item): item is [string, string] => 
                      Array.isArray(item) && 
                      item.length === 2 && 
                      typeof item[0] === 'string' && 
                      typeof item[1] === 'string'
                    );
                  setHistory(formattedHistory);
                }

                if (audioBase64) {
                  const binaryString = atob(audioBase64);
                  const bytes = new Uint8Array(binaryString.length);
                  bytes.set(Uint8Array.from(binaryString, c => c.charCodeAt(0)));
                  const audioBlob = new Blob([bytes], { type: "audio/mp3" });
                  audioManager.addToQueue(audioBlob);
                }
              } catch (error) {
                console.error("Error processing WebSocket message:", error);
              }
            };

            websocket.onclose = () => {
              console.log("WebSocket closed, reconnecting...");
              setTimeout(reconnectWebSocket, 5000);
            };

            websocket.onerror = (error) => {
              console.error("WebSocket error:", error);
              websocket?.close();
            };
          };

          reconnectWebSocket();
        } catch (error) {
          console.error("Error initializing:", error);
        }
      };

      initWebSocket();
    };

    document.body.appendChild(script);

    return () => {
      socket?.close();
    };
  }, [mediaRecorder, history, audioManager]);

  useEffect(() => {
    if (mediaRecorder?.state !== "inactive") {
      if (isRecording) {
        mediaRecorder?.resume();
      } else {
        mediaRecorder?.pause();
      }
    }
  }, [isRecording, mediaRecorder]);

  return (
    <>
      <div className={styles.title}>AudioChat - your voice AI assistant</div>
      <div className={styles["center-vertical"]}>
        <div
          className={`${styles["speaker-indicator"]} ${styles["you-speaking"]} ${isRecording && !isPlayingAudio ? styles.pulsate : ""}`}
        ></div>
        <br />
        <div>{isRecording && !isPlayingAudio ? "Listening..." : "Speaking..."}</div>
        <br />
        <div
          className={`${styles["speaker-indicator"]} ${styles["machine-speaking"]} ${!isRecording && isPlayingAudio ? styles.pulsate : ""}`}
        ></div>
      </div>
    </>
  );
}