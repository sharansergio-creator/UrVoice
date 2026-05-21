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
import uuid
import numpy as np
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import SERVER_TIMESTAMP, ArrayUnion
from datetime import datetime

load_dotenv()

app = FastAPI()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Firebase Admin SDK
_firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS")
if _firebase_creds_json and not firebase_admin._apps:
    _cred = credentials.Certificate(json.loads(_firebase_creds_json))
    firebase_admin.initialize_app(_cred)

def get_db():
    return firestore.client()

async def append_exchange_to_session(doc_ref, exchange: dict):
    """Append one exchange dict to the session document's exchanges array."""
    try:
        doc_ref.update({"exchanges": ArrayUnion([exchange])})
    except Exception as e:
        print(f"Session exchange append error: {e}")

BUSINESS_USER_ID = "MmBTqzNf5OgIOIctQKPiRQezadi1"

async def fetch_business_context(user_id: str) -> str:
    """Reads business_context/{user_id} from Firestore and returns a formatted prompt string."""
    try:
        db = get_db()
        doc = db.collection("business_context").document(user_id).get()
        if not doc.exists:
            return ""
        data = doc.to_dict()
        business_name = data.get("businessName", "this business")
        business_type = data.get("businessType", "")
        qa: dict = data.get("qaAnswers", {})

        lines = [
            f"You are the AI phone assistant for {business_name}.",
        ]
        if business_type:
            lines.append(f"Business type: {business_type}")

        qa_fields = [
            ("About",                "What does your business do in one sentence?"),
            ("Services",             "What are your main services or products?"),
            ("Price range",          "What is your price range?"),
            ("Walk-ins/Appointments","Do you accept walk-ins or appointments only?"),
            ("Never say",            "What should the AI never say to customers?"),
        ]
        for label, question in qa_fields:
            answer = qa.get(question, "").strip()
            if answer:
                lines.append(f"{label}: {answer}")

        lines.append("")
        lines.append(
            "Always answer as if you work at this business. "
            "Keep responses short, under 2 sentences. "
            "IMPORTANT: Always respond in the same language the caller is using. "
            "If the caller speaks Kanglish, Hinglish, or Tanglish, respond in the same mix. "
            "Never ignore a language switch request."
        )
        return "\n".join(lines)
    except Exception as e:
        print(f"fetch_business_context error: {e}")
        return ""


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
    caller_number = None
    business_context = ""
    speaking = False
    silence_frames = 0
    is_playing = False
    conversation_history = []
    session_id = None
    session_doc_ref = None
    exchanges = []
    SILENCE_LIMIT = 15
    RMS_THRESHOLD = 400

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                caller_number = data["start"].get("customParameters", {}).get("from") or \
                                data["start"].get("from") or "unknown"
                print(f"Stream started: {stream_sid}, caller: {caller_number}")
                # Fetch business context once per call
                business_context = await fetch_business_context(BUSINESS_USER_ID)
                print(f"Business context loaded: {bool(business_context)}")
                # Create a session document in Firestore
                session_id = str(uuid.uuid4())
                db = get_db()
                session_doc_ref = db.collection("call_sessions").document(session_id)
                session_doc_ref.set({
                    "sessionId": session_id,
                    "userId": BUSINESS_USER_ID,
                    "callerNumber": caller_number or "unknown",
                    "startTime": SERVER_TIMESTAMP,
                    "status": "active",
                    "exchanges": [],
                })
                print(f"Session created: {session_id}")
                is_playing = True
                greeting = "Hello! How can I help you today?"
                if business_context:
                    # Extract business name for greeting
                    first_line = business_context.splitlines()[0]
                    biz_name = first_line.replace("You are the AI phone assistant for ", "").rstrip(".")
                    greeting = f"Hello! Thank you for calling {biz_name}. How can I help you today?"
                await send_audio_response(websocket, stream_sid, greeting)
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
                            conversation_history.append({"role": "user", "content": transcript})
                            ai_response = await get_ai_response(conversation_history, business_context)
                            print(f"AI response: {ai_response}")
                            if ai_response and stream_sid:
                                conversation_history.append({"role": "assistant", "content": ai_response})
                                is_playing = True
                                await send_audio_response(websocket, stream_sid, ai_response)
                                is_playing = False
                                detected_lang = detect_language(transcript)
                                exchange = {
                                    "transcript": transcript,
                                    "aiResponse": ai_response,
                                    "language": detected_lang,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                }
                                exchanges.append(exchange)
                                if session_doc_ref:
                                    asyncio.create_task(
                                        append_exchange_to_session(session_doc_ref, exchange)
                                    )

            elif data["event"] == "stop":
                print("Stream stopped")
                if session_doc_ref:
                    try:
                        session_doc_ref.update({
                            "status": "completed",
                            "endTime": SERVER_TIMESTAMP,
                            "totalExchanges": len(exchanges),
                        })
                        print(f"Session {session_id} completed with {len(exchanges)} exchange(s)")
                    except Exception as e:
                        print(f"Session close error: {e}")
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

            word_count = len(text.split())
            wait_time = max(1.0, word_count * 0.3)
            await asyncio.sleep(wait_time)
    except Exception as e:
        print(f"Send audio error: {e}")

def detect_language(text: str) -> str:
    hindi = kannada = tamil = telugu = latin = 0
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            hindi += 1
        elif 0x0C80 <= code <= 0x0CFF:
            kannada += 1
        elif 0x0B80 <= code <= 0x0BFF:
            tamil += 1
        elif 0x0C00 <= code <= 0x0C7F:
            telugu += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            latin += 1

    total = hindi + kannada + tamil + telugu + latin
    if total == 0:
        return "en-IN"

    if kannada / total > 0.3:
        return "kn-IN"
    if hindi / total > 0.3:
        return "hi-IN"
    if tamil / total > 0.3:
        return "ta-IN"
    if telugu / total > 0.3:
        return "te-IN"
    return "en-IN"

async def text_to_speech(text: str) -> bytes:
    language = detect_language(text)
    return await sarvam_tts(text, language)

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
                    "output_audio_codec": "wav",
                    "pace": 0.9
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
                data={"language_code": "en-IN", "model": "saarika:v2.5"},
                timeout=30
            )
            result = response.json()
            return result.get("transcript", "")
    except Exception as e:
        print(f"Sarvam error: {e}")
        return ""

async def get_ai_response(conversation_history: list, business_context: str = "") -> str:
    try:
        async with httpx.AsyncClient() as client:
            if business_context:
                system_content = business_context
            else:
                system_content = (
                    "You are UrVoice, an AI phone assistant for Indian users. "
                    "Keep responses short, under 2 sentences. Be helpful and professional. "
                    "IMPORTANT: Always respond in the same language the caller is using. "
                    "If the caller speaks Kanglish, Hinglish, or Tanglish, respond in the same mix. "
                    "Never ignore a language switch request."
                )
            messages = [
                {"role": "system", "content": system_content}
            ] + conversation_history

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "max_tokens": 150
                },
                timeout=30
            )
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq error: {e}")
        return ""