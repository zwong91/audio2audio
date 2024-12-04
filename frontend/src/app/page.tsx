"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // 在组件顶部声明状态
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [currentAudioElement, setCurrentAudioElement] = useState<HTMLAudioElement | null>(null);

  // 音频管理函数
  const audioManager = {
    stopCurrentAudio: () => {
      if (currentAudioElement) {
        currentAudioElement.pause();
        currentAudioElement.currentTime = 0;
        URL.revokeObjectURL(currentAudioElement.src);
        setCurrentAudioElement(null);
        setIsPlayingAudio(false);
      }
    },

    playNewAudio: async (audioBlob: Blob) => {
      // 停止当前播放
      audioManager.stopCurrentAudio();

      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      setCurrentAudioElement(audio);
      setIsPlayingAudio(true);
      
      // 设置结束事件
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setCurrentAudioElement(null);
        setIsPlayingAudio(false);
        setIsRecording(true);
      };

      try {
        await audio.play();
      } catch (error) {
        console.error("播放音频失败:", error);
        audioManager.stopCurrentAudio();
      }
    }
  };


  // 首先定义 history 的类型
  type HistoryItem = [string, string]; // [用户输入, AI响应]
  type History = HistoryItem[];

  // 在组件中使用
  const [history, setHistory] = useState<History>([
    ['今天打老虎吗?', '没妞啊'],
    ['好久不见你还记得咱们大学那会儿吗你听到的是开项目 t t 那可是风华正茂的岁月啊还记得咱俩爬那个山顶看日初吗当时许多愿望我到现在还记得 😔', 
    '当然记得，那个时候真开心！一起爬山的事真的很怀念，你还记得许的愿望吗？']
  ]);
  const SOCKET_URL = "wss://gtp.aleopool.cc/stream";

  useEffect(() => {
    // Ensure screen stays awake
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

    // Clean up the wake lock on unmount
    return () => {
      if (wakeLock) {
        wakeLock.release().then(() => {
          console.log("Screen wake lock released");
        }).catch((error) => {
          console.error("Failed to release wake lock", error);
        });
      }
    };
  }, []); // Only run on mount and unmount

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        setMediaRecorder(new MediaRecorder(stream));
      }).catch((error) => {
        console.error("Error accessing media devices.", error);
      });
    } else {
      console.error("Media devices API not supported.");
    }
  }, []); // Setup mediaRecorder initially

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;
      let currentAudioElement: HTMLAudioElement | null = null; // Track the current playing audio element

      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          let websocket: WebSocket | null = null;

          // WebSocket reconnect logic
          const reconnectWebSocket = () => {
            if (websocket) websocket.close(); // Close existing WebSocket if it exists
            websocket = new WebSocket(SOCKET_URL);
            setSocket(websocket);

            websocket.onopen = () => {
              console.log("client connected to websocket");

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
                        // Convert ArrayBuffer to Base64
                        const base64data = arrayBufferToBase64(reader.result as ArrayBuffer);

                        // Prepare the data to be sent
                        const dataToSend = [
                          history, // Include the stored history
                          "xiaoxiao", // The user identifier or other identifier
                          base64data // The base64 encoded audio data
                        ];
                        const jsonData = JSON.stringify(dataToSend);

                        // Safe check to ensure websocket is not null
                        if (websocket) {
                          websocket.send(jsonData);
                        } else {
                          console.error("WebSocket is null, cannot send data.");
                        }
                      } else {
                        console.error("FileReader result is null");
                      }
                    };
                    reader.readAsArrayBuffer(blob); // Read as ArrayBuffer
                  }
                }
              });

              recorder.startRecording();
            };

            websocket.onmessage = (event) => {
              setIsRecording(false); // Stop recording when receiving message
              setIsPlayingAudio(true); // Start playing audio
            
              try {
                const jsonData = JSON.parse(event.data);
                const audioBase64 = jsonData["stream"];
                
                const receivedHistory = jsonData["history"]; // Extract the history
                if (Array.isArray(receivedHistory)) {
                  // 确保收到的历史记录是二维数组结构
                  const formattedHistory = receivedHistory.map(item => 
                    Array.isArray(item) ? item : [item[0], item[1]]
                  );
                  setHistory(formattedHistory);
                }
                if (!audioBase64) {
                  console.error("No audio stream data received");
                  return;
                }

              // 转换音频数据
              const binaryString = atob(audioBase64);
              const bytes = new Uint8Array(binaryString.length);
              bytes.set(Uint8Array.from(binaryString, c => c.charCodeAt(0)));
              const audioBlob = new Blob([bytes], { type: "audio/wav" });

              // 播放新音频
              audioManager.playNewAudio(audioBlob);
            
              } catch (error) {
                console.error("Error processing WebSocket message:", error);
              }
            };

            websocket.onclose = () => {
              console.log("WebSocket connection closed, attempting to reconnect...");
              setTimeout(reconnectWebSocket, 5000); // Retry after 5 seconds
            };

            websocket.onerror = (error) => {
              console.error("WebSocket error:", error);
              websocket?.close();
            };
          };

          reconnectWebSocket(); // Initial connection attempt
        }).catch((error) => {
          console.error("Error with getUserMedia", error);
        });
      }
    };
    document.body.appendChild(script);

    // Cleanup on component unmount
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [mediaRecorder]);

  useEffect(() => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      if (isRecording) {
        mediaRecorder.resume();
      } else {
        mediaRecorder.pause();
      }
    }
  }, [isRecording, mediaRecorder]);

  // Helper function to convert ArrayBuffer to Base64
  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    let binary = '';
    const uint8Array = new Uint8Array(arrayBuffer);
    const len = uint8Array.length;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary); // Convert binary string to base64
  }

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
