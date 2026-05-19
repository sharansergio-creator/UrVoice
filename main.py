from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import base64
import httpx
import audioop
import wave
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

@app.get("/")
def root():
    return {"status": "UrVoice backend running"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host")
    response = VoiceResponse()
    response.say("Hello, you have reached UrVoice. Please speak after the beep.", voice="alice")
    response.pause(length=1)
    connect = Connect()
    connect.stream(url=f"wss://{host}/audio-stream")
    response.append(connect)
    return PlainTextResponse(str(response), media_type="application/xml")

@app.websocket("/audio-stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    audio_chunks = []

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["event"] == "media":
                audio_chunks.append(data["media"]["payload"])

            elif data["event"] == "stop":
                if audio_chunks:
                    raw_mulaw = b"".join(base64.b64decode(chunk) for chunk in audio_chunks)
                    wav_bytes = mulaw_to_wav(raw_mulaw)
                    transcript = await transcribe(wav_bytes)
                    print(f"Caller said: {transcript}")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")

def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    pcm = audioop.ulaw2lin(mulaw_bytes, 2)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    return buf.getvalue()

async def transcribe(audio_bytes: bytes) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": "unknown", "model": "saaras:v2"},
                timeout=30
            )
            print(f"Sarvam response: {response.status_code} {response.text}")
            result = response.json()
            return result.get("transcript", "")
    except Exception as e:
        print(f"Sarvam error: {e}")
        return ""