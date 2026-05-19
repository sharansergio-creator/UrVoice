from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import base64
import httpx
import wave
import io
import asyncio
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

@app.get("/")
def root():
    return {"status": "UrVoice backend running"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/audio-stream")
    response.append(connect)
    return PlainTextResponse(str(response), media_type="application/xml")

@app.websocket("/audio-stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    audio_chunks = []
    stream_sid = None
    speaking = False
    silence_frames = 0
    is_playing = False
    SILENCE_LIMIT = 25
    RMS_THRESHOLD = 300

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                print(f"Stream started: {stream_sid}")
                is_playing = True
                await send_audio_response(
                    websocket, stream_sid,
                    "Hello! You have reached UrVoice. How can I help you today?"
                )
                is_playing = False

            elif data["event"] == "media":
                if is_playing:
                    continue

                raw_chunk = base64.b64decode(data["media"]["payload"])
                audio_chunks.append(data["media"]["payload"])

                pcm_chunk = mulaw_chunk_to_pcm(raw_chunk)
                samples = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(samples ** 2))

                if rms > RMS_THRESHOLD:
                    speaking = True
                    silence_frames = 0
                elif speaking:
                    silence_frames += 1
                    if silence_frames >= SILENCE_LIMIT:
                        speaking = False
                        silence_frames = 0
                        chunks_to_process = audio_chunks.copy()
                        audio_chunks.clear()

                        raw_mulaw = b"".join(base64.b64decode(c) for c in chunks_to_process)
                        wav_bytes = mulaw_to_wav(raw_mulaw)
                        transcript = await transcribe(wav_bytes)
                        print(f"Caller said: {transcript}")

                        if transcript and transcript.strip():
                            ai_response = await get_ai_response(transcript)
                            print(f"AI response: {ai_response}")
                            if ai_response and stream_sid:
                                is_playing = True
                                await send_audio_response(websocket, stream_sid, ai_response)
                                is_playing = False

            elif data["event"] == "stop":
                print("Stream stopped")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")

async def send_audio_response(websocket: WebSocket, stream_sid: str, text: str):
    try:
        audio_bytes = await text_to_speech(text)
        if audio_bytes:
            mulaw_audio = pcm_to_mulaw(audio_bytes)
            payload = base64.b64encode(mulaw_audio).decode("utf-8")
            message = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            }
            await websocket.send_text(json.dumps(message))
            print(f"Sent audio response for: {text[:50]}")

            # Wait for audio to finish playing before accepting new input
            word_count = len(text.split())
            wait_time = max(1.5, word_count * 0.4)
            await asyncio.sleep(wait_time)
    except Exception as e:
        print(f"Send audio error: {e}")

def detect_language(text: str) -> str:
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "hi-IN"
        if 0x0C80 <= code <= 0x0CFF:
            return "kn-IN"
        if 0x0B80 <= code <= 0x0BFF:
            return "ta-IN"
        if 0x0C00 <= code <= 0x0C7F:
            return "te-IN"
    return "en"

async def text_to_speech(text: str) -> bytes:
    language = detect_language(text)
    if language == "en":
        return await cartesia_tts(text)
    else:
        return await sarvam_tts(text, language)

async def cartesia_tts(text: str) -> bytes:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "Cartesia-Version": "2024-06-10",
                    "X-API-Key": CARTESIA_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "transcript": text,
                    "model_id": "sonic-english",
                    "voice": {
                        "mode": "id",
                        "id": "79a125e8-cd45-4c13-8a67-188112f4dd22"
                    },
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": 8000
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.content
            else:
                print(f"Cartesia error: {response.status_code} {response.text}")
                return None
    except Exception as e:
        print(f"Cartesia TTS error: {e}")
        return None

async def sarvam_tts(text: str, language_code: str) -> bytes:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "inputs": [text],
                    "target_language_code": language_code,
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                    "speech_sample_rate": 8000,
                    "enable_preprocessing": True,
                    "output_audio_codec": "wav"
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                audio_b64 = result["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)
                buf = io.BytesIO(audio_bytes)
                with wave.open(buf, 'rb') as wf:
                    pcm_bytes = wf.readframes(wf.getnframes())
                return pcm_bytes
            else:
                print(f"Sarvam TTS error: {response.status_code} {response.text}")
                return None
    except Exception as e:
        print(f"Sarvam TTS error: {e}")
        return None

def pcm_to_mulaw(pcm_bytes: bytes) -> bytes:
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
    sign = np.where(samples < 0, 0x80, 0x00)
    samples = np.abs(samples)
    samples = np.clip(samples, 0, 32767)
    samples = samples + MULAW_BIAS
    samples = np.clip(samples, 0, MULAW_MAX)

    exp = np.zeros(len(samples), dtype=np.int32)
    for i in range(7, -1, -1):
        mask = samples >= (1 << (i + 5))
        exp = np.where(mask & (exp == 0), i, exp)

    mantissa = (samples >> (exp + 1)) & 0x0F
    mulaw = ~(sign | (exp << 4) | mantissa)
    return (mulaw & 0xFF).astype(np.uint8).tobytes()

def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    mulaw_array = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    mulaw_array = mulaw_array.astype(np.int32)
    mulaw_array = ~mulaw_array
    sign = mulaw_array & 0x80
    exponent = (mulaw_array >> 4) & 0x07
    mantissa = mulaw_array & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample = np.where(sign != 0, 0x84 - sample, sample - 0x84)
    pcm = sample.astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    return buf.getvalue()

def mulaw_chunk_to_pcm(mulaw_bytes: bytes) -> bytes:
    mulaw_array = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    mulaw_array = mulaw_array.astype(np.int32)
    mulaw_array = ~mulaw_array
    sign = mulaw_array & 0x80
    exponent = (mulaw_array >> 4) & 0x07
    mantissa = mulaw_array & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample = np.where(sign != 0, 0x84 - sample, sample - 0x84)
    return sample.astype(np.int16).tobytes()

async def transcribe(audio_bytes: bytes) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": "unknown", "model": "saarika:v2.5"},
                timeout=30
            )
            result = response.json()
            return result.get("transcript", "")
    except Exception as e:
        print(f"Sarvam error: {e}")
        return ""

async def get_ai_response(transcript: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are UrVoice, an AI phone assistant. Keep responses short, under 2 sentences. Be helpful and professional. Only ask the caller to repeat if the message is completely unintelligible gibberish. Short phrases, single words, and partial sentences are valid — respond to them normally."
                        },
                        {
                            "role": "user",
                            "content": transcript
                        }
                    ],
                    "max_tokens": 150
                },
                timeout=30
            )
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq error: {e}")
        return ""