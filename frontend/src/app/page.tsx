"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import styles from "./page.module.css";

export default function VoiceCall() {
  // 状态管理
  const [isRecording, setIsRecording] = useState(true);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<string>("连接中...");
  const [callDuration, setCallDuration] = useState<number>(0);
  const [networkStatus, setNetworkStatus] = useState<boolean>(navigator.onLine);
  const [wakeLock, setWakeLock] = useState<WakeLockSentinel | null>(null);

  // Refs
  const socketRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioBufferQueue = useRef<AudioBuffer[]>([]);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>();

  // 初始化 AudioContext
  useEffect(() => {
    if (typeof window !== "undefined" && window.AudioContext) {
      audioContextRef.current = new AudioContext();
    }
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // 通话计时器
  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // 时间格式化
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // 音频播放队列处理
  const playAudioBufferQueue = useCallback(() => {
    if (audioBufferQueue.current.length === 0) {
      setIsPlayingAudio(false);
      setIsRecording(true);
      return;
    }

    const buffer = audioBufferQueue.current.shift();
    if (buffer && audioContextRef.current) {
      const source = audioContextRef.current.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContextRef.current.destination);
      source.onended = playAudioBufferQueue;
      source.start();
      setIsPlayingAudio(true);
    }
  }, []);

  // 音频缓冲处理
  const bufferAudio = useCallback((data: ArrayBuffer) => {
    if (audioContextRef.current) {
      audioContextRef.current.decodeAudioData(data, (buffer) => {
        audioBufferQueue.current.push(buffer);
        if (!isPlayingAudio) {
          playAudioBufferQueue();
        }
      });
    }
  }, [isPlayingAudio, playAudioBufferQueue]);

  // WebSocket 初始化
  const initWebSocket = useCallback(
    (mediaStream: MediaStream, RecordRTC: any, StereoAudioRecorder: any) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.close();
      }

      const SOCKET_URL = "wss://gtp.aleopool.cc/stream";
      const websocket = new WebSocket(SOCKET_URL);
      socketRef.current = websocket;
      websocket.binaryType = "arraybuffer";

      websocket.onopen = () => {
        setConnectionStatus("已连接");
        
        // 配置录音器
        const newRecorder = new RecordRTC(mediaStream, {
          type: 'audio',
          mimeType: 'audio/wav',
          recorderType: StereoAudioRecorder,
          timeSlice: 500,
          desiredSampRate: 16000,
          numberOfAudioChannels: 1,
          bufferSize: 4096,
          disableLogs: true,
          ondataavailable: (blob: Blob) => {
            if (!blob || blob.size === 0) return;
            
            const reader = new FileReader();
            reader.onloadend = () => {
              if (reader.result && websocket.readyState === WebSocket.OPEN) {
                try {
                  const base64data = arrayBufferToBase64(reader.result as ArrayBuffer);
                  const dataToSend = [[], "xiaoxiao", base64data];
                  websocket.send(JSON.stringify(dataToSend));
                } catch (error) {
                  console.error('发送音频数据失败:', error);
                }
              }
            };
            reader.readAsArrayBuffer(blob);
          }
        });

        try {
          newRecorder.startRecording();
          recorderRef.current = newRecorder;

        } catch (error) {
          console.error('启动录音失败:', error);
          setConnectionStatus('录音启动失败');
        }
      };

      websocket.onmessage = (event: MessageEvent) => {
        try {
          // 处理音频数据
          setIsRecording(false);
          setIsPlayingAudio(true);
      
          // 处理二进制数据
          const handleBinaryData = (binaryData: ArrayBuffer) => {
            try {
              bufferAudio(binaryData);
            } catch (error) {
              console.error("处理音频数据失败:", error);
            }
          };
      
          if (event.data instanceof ArrayBuffer) {
            handleBinaryData(event.data);
          } else if (event.data instanceof Blob) {
            const reader = new FileReader();
            reader.onloadend = () => {
              if (reader.result) {
                handleBinaryData(reader.result as ArrayBuffer);
              }
            };
            reader.onerror = (error) => {
              console.error("读取Blob数据失败:", error);
            };
            reader.readAsArrayBuffer(event.data);
          } else {
            console.warn("收到未知类型的消息:", typeof event.data);
          }
        } catch (error) {
          console.error("处理WebSocket消息失败:", error);
        }
      };

      websocket.onclose = () => {
        console.log("WebSocket连接已断开，正在重连...");
        setConnectionStatus("重新连接中...");
        if (recorderRef.current) {
          recorderRef.current.stopRecording();
        }
        
        reconnectTimeoutRef.current = setTimeout(() => {
          initWebSocket(mediaStream, RecordRTC, StereoAudioRecorder);
        }, 5000);
      };

      websocket.onerror = (error: Event) => {
        console.error("WebSocket错误:", error);
        websocket.close();
      };
    },
    [bufferAudio]
  );

  // 初始化
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";

    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;

      navigator.mediaDevices
        ?.getUserMedia({ audio: true })
        .then((mediaStream) => {
          streamRef.current = mediaStream;
          initWebSocket(mediaStream, RecordRTC, StereoAudioRecorder);
        })
        .catch((error) => console.error("麦克风访问失败:", error));
    };

    document.body.appendChild(script);

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (recorderRef.current) {
        recorderRef.current.stopRecording();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      document.body.removeChild(script);
    };
  }, [initWebSocket]);

  // 网络状态监控
  useEffect(() => {
    const handleNetworkChange = () => {
      setNetworkStatus(navigator.onLine);
    };

    window.addEventListener("online", handleNetworkChange);
    window.addEventListener("offline", handleNetworkChange);

    return () => {
      window.removeEventListener("online", handleNetworkChange);
      window.removeEventListener("offline", handleNetworkChange);
    };
  }, []);

  // 屏幕常亮
  useEffect(() => {
    const requestWakeLock = async () => {
      try {
        const wl = await navigator.wakeLock.request("screen");
        setWakeLock(wl);
      } catch (err) {
        console.error("屏幕常亮启用失败:", err);
      }
    };

    requestWakeLock();

    const handleVisibilityChange = async () => {
      if (document.visibilityState === "visible" && !wakeLock) {
        await requestWakeLock();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      wakeLock?.release().catch(console.error);
    };
  }, [wakeLock]);

  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    const uint8Array = new Uint8Array(arrayBuffer);
    let binary = "";
    uint8Array.forEach((byte) => (binary += String.fromCharCode(byte)));
    return btoa(binary);
  }

  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div
            className={`${styles.statusDot} ${
              connectionStatus === "已连接" ? styles.connected : ""
            }`}
          />
          {connectionStatus}
        </div>
        <div className={styles.duration}>{formatTime(callDuration)}</div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.avatarSection}>
          <div
            className={`${styles.avatarContainer} ${
              isPlayingAudio ? styles.speaking : ""
            }`}
          >
            <img src="/ai-avatar.png" alt="AI" className={styles.avatar} />
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