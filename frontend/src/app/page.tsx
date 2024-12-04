"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback

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

      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          const socket = new WebSocket("wss://gtp.aleopool.cc/stream");

          socket.onopen = () => {
            console.log("client connected to websocket");

            const recorder = new RecordRTC(stream, {
              type: 'audio',
              recorderType: StereoAudioRecorder,
              mimeType: 'audio/wav',
              timeSlice: 250,
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
                        [], // Empty array, could be used for other data
                        "xiaoxiao", // The user identifier or other identifier
                        base64data // The base64 encoded audio data
                      ];
                      const jsonData = JSON.stringify(dataToSend);
                      socket.send(jsonData);
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

          socket.onmessage = (event) => {
            setIsRecording(false); // Stop recording when receiving message
            setIsPlayingAudio(true); // Start playing audio
            
            try {
              const jsonData = JSON.parse(event.data);
              const audioBase64 = jsonData["stream"];

              if (!audioBase64) {
                console.error("No audio stream data received");
                return;
              }

              // Convert Base64 to Audio Blob
              const binaryString = atob(audioBase64);
              const len = binaryString.length;
              const bytes = new Uint8Array(len);
              for (let i = 0; i < len; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }

              const blob = new Blob([bytes], { type: "audio/mp3" });

              // Play the received audio
              const audioUrl = URL.createObjectURL(blob);
              const audioElement = new Audio(audioUrl);

              // Listen for when the audio finishes playing
              audioElement.onended = () => {
                setIsPlayingAudio(false); // Finished playing, stop audio playback state
                setIsRecording(true); // Resume recording after playback
                URL.revokeObjectURL(audioUrl); // Clean up URL
              };

              audioElement.play().catch((error) => {
                console.error("Error playing audio:", error);
              });

            } catch (error) {
              console.error("Error processing WebSocket message:", error);
            }
          };

          // Clean up the socket connection on component unmount
          return () => {
            socket.close();
          };
        }).catch((error) => {
          console.error("Error with getUserMedia", error);
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
