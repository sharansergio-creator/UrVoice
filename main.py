from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import base64
import httpx
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
                    # Combine all audio chunks
                    combined = b"".join(base64.b64decode(chunk) for chunk in audio_chunks)
                    
                    # Send to Sarvam STT
                    transcript = await transcribe(combined)
                    print(f"Caller said: {transcript}")
                
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")

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
            result = response.json()
            return result.get("transcript", "")
    except Exception as e:
        print(f"Sarvam error: {e}")
        return ""