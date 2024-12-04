"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true);

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

      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          const socket = new WebSocket("wss://gtp.aleopool.cc/stream");

          socket.onopen = () => {
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
                      const base64data = (reader.result as string).split(',')[1];
                      const data_to_send = [
                        [],
                        "xiaoxiao",
                        base64data
                      ];
                      const json_data = JSON.stringify(data_to_send);
                      socket.send(json_data);
                    } else {
                      console.error("FileReader result is null");
                    }
                  };
                  reader.readAsDataURL(blob);
                }
              }
            });

            recorder.startRecording();
          };

          socket.onmessage = (event) => {
            setIsRecording(false);
            const received = event.data;
            // 解析接收到的 JSON 数据
            const jsonData = JSON.parse(received);
            const history = jsonData["history"];
            const audio = jsonData["audio"];
            const text = jsonData["text"];
            
            const audioBase64 = jsonData["stream"];
            const binaryString = atob(audioBase64);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            
            for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            const audioArrayBuffer = bytes.buffer;
      
            const blob = new Blob([audioArrayBuffer], { type: "audio/mpeg" });
            const audioUrl = URL.createObjectURL(blob);

            const audioElement = new Audio(audioUrl);
            audioElement.onended = () => setIsRecording(true);
            audioElement.play();
          };

          return () => {
            socket.close(); // Clean up the socket connection on component unmount
          };
        });
      }
    };
    document.body.appendChild(script);
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

  return (
    <>
      <div className={styles.title}>AudioChat - your voice AI assistant</div>
      <div className={styles["center-vertical"]}>
        <div
          className={`${styles["speaker-indicator"]} ${styles["you-speaking"]} ${isRecording ? styles.pulsate : ""}`}
        ></div>
        <br />
        <div>{isRecording ? "Listening..." : "Speaking..."}</div>
        <br />
        <div
          className={`${styles["speaker-indicator"]} ${styles["machine-speaking"]} ${!isRecording ? styles.pulsate : ""}`}
        ></div>
      </div>
    </>
  );
}
