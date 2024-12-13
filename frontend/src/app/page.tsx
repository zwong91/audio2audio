"use client";
import { useEffect, useState } from "react";
import styles from "./page.module.css";
export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<string>("Connecting..."); // State to track connection status
  let audioContext: AudioContext | null = null;
  let audioBufferQueue: AudioBuffer[] = [];
  // Check if AudioContext is available in the browser
  if (typeof window !== "undefined" && window.AudioContext) {
    audioContext = new AudioContext();
  }

  // 新增通话时长状态
  const [callDuration, setCallDuration] = useState<number>(0);

  // 添加通话计时器
  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);


  // Check if AudioContext is available in the browser
  if (typeof window !== "undefined" && window.AudioContext) {
    audioContext = new AudioContext();
  }

  const audioManager = {
    stopCurrentAudio: () => {
      if (isPlayingAudio) {
        setIsPlayingAudio(false);
      }
    },

    playNewAudio: async (audioBlob: Blob) => {
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      // When the metadata of the audio is loaded, set its duration
      audio.onloadedmetadata = () => {
        setAudioDuration(audio.duration); // Set the audio duration after loading metadata
      };

      // Play the audio
      setIsPlayingAudio(true);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsPlayingAudio(false);
        setIsRecording(true);

        // 延迟 0.1 秒再进行操作
        setTimeout(() => {
          if (audioQueue.length > 0) {
            const nextAudioBlob = audioQueue.shift();
            if (nextAudioBlob) {
              audioManager.playNewAudio(nextAudioBlob); // Play next audio in the queue
            }
          }
        }, 100); // 延迟 0.1 秒再进行操作
      };

      try {
        await audio.play();
      } catch (error) {
        console.error("播放音频失败:", error);
        audioManager.stopCurrentAudio();
      }
    }
  };

  // Buffer audio and add it to the queue
  function bufferAudio(data: ArrayBuffer) {
    if (audioContext) {
      audioContext.decodeAudioData(data, (buffer) => {
        // Buffer the audio chunk and push it to the queue
        audioBufferQueue.push(buffer);

        // If we are not already playing, start playing the audio
        if (!isPlayingAudio) {
          playAudioBufferQueue();
        }
      });
    }
  }

  // Play the buffered audio chunks from the queue
  function playAudioBufferQueue() {
    if (audioBufferQueue.length === 0) {
      setIsPlayingAudio(false); // Stop playback if queue is empty
      setIsRecording(true); // Start recording again
      return;
    }

    const buffer = audioBufferQueue.shift(); // Get the next audio buffer
    if (buffer && audioContext) {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;

      // Connect the source to the audio context's output
      source.connect(audioContext.destination);

      // When this audio ends, play the next one
      source.onended = () => {
        playAudioBufferQueue(); // Continue playing the next buffer
      };

      // Start playing the audio
      source.start();

      // Update the state to reflect the playing status
      setIsPlayingAudio(true);
    }
  }


  const SOCKET_URL = "wss://gtp.aleopool.cc/stream";
  // Initialize WebSocket and media devices
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;
    // Request screen wake lock to prevent the screen from going to sleep
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
      if (wakeLock) {
        wakeLock.release().then(() => {
          console.log("Screen wake lock released");
        }).catch((error) => {
          console.error("Failed to release wake lock", error);
        });
      }
    };
  }, []);
  // Access the microphone and start recording
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
  }, []);
  // Handle WebSocket connection and messaging
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          let websocket: WebSocket | null = null;
          const reconnectWebSocket = () => {
            if (websocket) websocket.close();
            websocket = new WebSocket(SOCKET_URL);
            setSocket(websocket);
            websocket.onopen = () => {
              console.log("client connected to websocket");
              setConnectionStatus("Connected");
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
                        const dataToSend = [
                          [],
                          "xiaoxiao",
                          base64data
                        ];
                        const jsonData = JSON.stringify(dataToSend);
                        if (websocket) {
                          websocket.send(jsonData);
                        } else {
                          console.error("WebSocket is null, cannot send data.");
                        }
                      } else {
                        console.error("FileReader result is null");
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
                let audioData: ArrayBuffer;

                // 如果 event.data 是 ArrayBuffer，直接处理
                if (event.data instanceof ArrayBuffer) {
                  audioData = event.data;
                } else if (event.data instanceof Blob) {
                  // 如果是 Blob 类型，使用 FileReader 将其转换为 ArrayBuffer
                  const reader = new FileReader();
                  reader.onloadend = () => {
                    audioData = reader.result as ArrayBuffer;
                    bufferAudio(audioData); // Buffer the audio
                  };
                  reader.readAsArrayBuffer(event.data);
                  return;
                } else {
                  throw new Error("Received unexpected data type from WebSocket");
                }

                bufferAudio(audioData);
              } catch (error) {
                console.error("Error processing WebSocket message:", error);
              }
            };
            websocket.onclose = () => {
              console.log("WebSocket connection closed, attempting to reconnect...");
              setConnectionStatus("Reconnecting...");
              setTimeout(reconnectWebSocket, 5000);
            };
            websocket.onerror = (error) => {
              console.error("WebSocket error:", error);
              websocket?.close();
            };
          };
          reconnectWebSocket();
        }).catch((error) => {
          console.error("Error with getUserMedia", error);
        });
      }
    };
    document.body.appendChild(script);
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);
  // Handle media recorder pause/resume
  useEffect(() => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      if (isRecording) {
        mediaRecorder.resume();
      } else {
        mediaRecorder.pause();
      }
    }
  }, [isRecording, mediaRecorder]);
  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    let binary = '';
    const uint8Array = new Uint8Array(arrayBuffer);
    const len = uint8Array.length;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
  }
  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div
            className={`${styles.statusDot} ${
              connectionStatus === "Connected" ? styles.connected : ""
            }`}
          />
          {connectionStatus}
        </div>
        <div className={styles.duration}>{callDuration}s</div>
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