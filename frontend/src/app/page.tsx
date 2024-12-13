// page.tsx
"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import styles from "./page.module.css";

export default function VoiceCall() {
  const [isRecording, setIsRecording] = useState(true);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<string>("连接中...");
  const [callDuration, setCallDuration] = useState<number>(0);
  const [networkStatus, setNetworkStatus] = useState<boolean>(navigator.onLine);
  const [wakeLock, setWakeLock] = useState<WakeLockSentinel | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);

  let audioContext: AudioContext | null = null;
  let audioBufferQueue: AudioBuffer[] = [];

  if (typeof window !== "undefined" && window.AudioContext) {
    audioContext = new AudioContext();
  }

  // 通话计时器
  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
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
          if (audioBufferQueue.length > 0) {
            playAudioBufferQueue();
          }
        }, 100);
      };

      try {
        await audio.play();
      } catch (error) {
        console.error("播放失败:", error);
        audioManager.stopCurrentAudio();
      }
    },
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

  const handleAudioData = useCallback((blob: Blob) => {
    if (blob.size > 0) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result) {
          const base64data = arrayBufferToBase64(
            reader.result as ArrayBuffer
          );
          const dataToSend = [[], "xiaoxiao", base64data];
          socketRef.current?.send(JSON.stringify(dataToSend));
        }
      };
      reader.readAsArrayBuffer(blob);
    }
  }, []);

  const initWebSocket = useCallback(
    (mediaStream: MediaStream, RecordRTC: any, StereoAudioRecorder: any) => {
      const SOCKET_URL = "wss://gtp.aleopool.cc/stream";
      const websocket = new WebSocket(SOCKET_URL);
      socketRef.current = websocket;

      websocket.binaryType = "arraybuffer"; // 确保接收的是二进制数据

      websocket.onopen = () => {
        setConnectionStatus("已连接");

        // 在 WebSocket 连接成功后启动录音器
        const newRecorder = new RecordRTC(mediaStream, {
          type: "audio",
          recorderType: StereoAudioRecorder,
          mimeType: "audio/wav",
          timeSlice: 500, // 每隔500ms触发ondataavailable
          desiredSampRate: 16000,
          numberOfAudioChannels: 1,
          ondataavailable: handleAudioData,
        });

        newRecorder.startRecording();
        recorderRef.current = newRecorder;
      };

      websocket.onmessage = (event: MessageEvent) => {
        setIsRecording(false);
        setIsPlayingAudio(true);

        try {
          let audioData: ArrayBuffer;
          if (event.data instanceof ArrayBuffer) {
            audioData = event.data;
            bufferAudio(audioData);
          } else if (event.data instanceof Blob) {
            const reader = new FileReader();
            reader.onloadend = () => {
              audioData = reader.result as ArrayBuffer;
              bufferAudio(audioData);
            };
            reader.readAsArrayBuffer(event.data);
          } else {
            throw new Error("未知的数据类型");
          }
        } catch (error) {
          console.error("音频处理失败:", error);
        }
      };

      websocket.onclose = () => {
        console.log("WebSocket连接已断开，正在重连...");
        setConnectionStatus("重新连接中...");
        recorderRef.current?.stopRecording();
        setTimeout(() => {
          initWebSocket(mediaStream, RecordRTC, StereoAudioRecorder);
        }, 5000);
      };

      websocket.onerror = (error: Event) => {
        console.error("WebSocket错误:", error);
        websocket.close();
      };
    },
    [handleAudioData]
  );

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
      socketRef.current?.close();
      recorderRef.current?.stopRecording();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [initWebSocket]);

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

  useEffect(() => {
    const requestWakeLock = async () => {
      try {
        const wl = await navigator.wakeLock.request("screen");
        setWakeLock(wl);
        console.log("屏幕常亮已启用");
      } catch (err) {
        console.error("屏幕常亮启用失败:", err);
      }
    };

    requestWakeLock();

    const handleVisibilityChange = async () => {
      if (document.visibilityState === "visible" && !wakeLock) {
        try {
          const wl = await navigator.wakeLock.request("screen");
          setWakeLock(wl);
        } catch (err) {
          console.error("重新激活屏幕常亮失败:", err);
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener(
        "visibilitychange",
        handleVisibilityChange
      );
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
            <div className={styles.audioWaves}>
              {Array(3)
                .fill(0)
                .map((_, i) => (
                  <div
                    key={i}
                    className={`${styles.wave} ${
                      isPlayingAudio ? styles.active : ""
                    }`}
                  />
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