from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse, Connect
import os
import json
import base64
import httpx
import wave
import io
import asyncio
import uuid
import re
import numpy as np
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import SERVER_TIMESTAMP, ArrayUnion
from datetime import datetime
from bs4 import BeautifulSoup

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
    try:
        db = get_db()
        doc = db.collection("business_context").document(user_id).get()
        if not doc.exists:
            return ""
        data = doc.to_dict()
        return build_system_prompt(data)
    except Exception as e:
        print(f"fetch_business_context error: {e}")
        return ""


def build_system_prompt(data: dict) -> str:
    sections = []

    # Identity
    name = data.get("businessName", "this business")
    btype = data.get("businessType", "business")
    location = data.get("location", "")
    sections.append(
        f"You are the AI phone assistant for {name}, a {btype}"
        + (f", located in {location}" if location else "") + "."
    )

    # Contact
    phone = data.get("phone", "")
    email = data.get("email", "")
    if phone or email:
        contact = f"Contact: {phone}"
        if email:
            contact += f", {email}"
        sections.append(contact)

    # About
    about = data.get("about", "")
    if about:
        sections.append(f"About: {about}")

    # Services
    services = data.get("services", "")
    if services:
        sections.append(f"Services: {services}")

    # Accommodations (for resorts/hotels)
    accommodations = data.get("accommodations", "")
    if accommodations:
        sections.append(f"Accommodation options: {accommodations}")

    # Activities (for resorts)
    activities = data.get("activities", "")
    if activities:
        sections.append(f"Activities: {activities}")

    # Paid activities
    paid_activities = data.get("paidActivities", "")
    if paid_activities:
        sections.append(f"Extra paid activities: {paid_activities}")

    # Pricing
    pricing = data.get("pricing", "")
    if pricing:
        sections.append(f"Pricing: {pricing}")

    # Hours
    hours = data.get("hours", "")
    if hours:
        sections.append(f"Business hours: {hours}")

    # Events
    events = data.get("events", "")
    if events:
        sections.append(f"Events hosted: {events}")

    # Social media
    instagram = data.get("instagram", "")
    if instagram:
        sections.append(f"Instagram: {instagram}")

    facebook = data.get("facebook", "")
    if facebook:
        sections.append(f"Facebook: {facebook}")

    # OTA platforms
    ota = data.get("otaPlatforms", "")
    if ota:
        sections.append(f"Listed on: {ota}")

    # QA answers from BusinessSetup screen
    qa = data.get("qaAnswers", {})
    qa_fields = [
        ("What does your business do in one sentence?", "About"),
        ("What are your main services or products?", "Services"),
        ("What is your price range?", "Price range"),
        ("Do you accept walk-ins or appointments only?", "Bookings"),
        ("What should the AI never say to customers?", "Never say"),
    ]
    existing = "\n".join(sections)
    for question, label in qa_fields:
        answer = qa.get(question, "").strip()
        if answer and label not in existing:
            sections.append(f"{label}: {answer}")

    # Website
    website = data.get("websiteUrl", "")
    if website:
        sections.append(f"Website: {website}")

    # Universal rules
    sections.append("""
Rules:
- Always respond as a helpful, friendly staff member of this business
- Keep responses short, under 2 sentences
- Always respond in the same language the caller uses
- Never make up pricing or availability — say you will check and confirm
- For bookings, always ask for date, number of people, and preference
- If you don't know something, say you will check and call back
- Never ignore a language switch request from the caller""")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# /fetch-business-context helpers
# ---------------------------------------------------------------------------

class FetchBusinessContextRequest(BaseModel):
    gbp_url: str = ""
    website_url: str = ""
    user_id: str = BUSINESS_USER_ID


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def _fetch_html(url: str, follow_redirects: bool = True) -> str:
    """Fetch a URL and return the decoded HTML body."""
    async with httpx.AsyncClient(
        headers=_BROWSER_HEADERS,
        follow_redirects=follow_redirects,
        timeout=15,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _jina_read(url: str) -> str:
    """Convert any URL to clean LLM-readable text via Jina Reader.
    Falls back to direct httpx + BeautifulSoup if Jina fails/times out."""
    # Try Jina first (best for JS-rendered pages)
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _BROWSER_HEADERS["User-Agent"], "X-Return-Format": "text"},
            follow_redirects=True,
            timeout=20,
        ) as client:
            resp = await client.get(f"https://r.jina.ai/{url}")
            if resp.status_code == 200 and len(resp.text.strip()) > 150:
                return resp.text[:8000]
            print(f"Jina returned {resp.status_code} / short response for {url}")
    except Exception as e:
        print(f"Jina read error for {url}: {e}")

    # Fallback: direct httpx + BeautifulSoup
    try:
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS, follow_redirects=True, timeout=12
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                for tag in soup.select("nav, header, footer, script, style, noscript"):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)
                # Collapse whitespace
                text = re.sub(r"\s{3,}", "  ", text)
                return text[:8000]
    except Exception as e:
        print(f"Direct fetch error for {url}: {e}")
    return ""


async def _scrape_gbp(gbp_url: str) -> dict:
    """Extract business info from a Google Business Profile URL.
    Jina blocks google.com so we extract the name from the URL and then
    try a direct HTML fetch for any meta-data."""
    result = {"businessName": "", "address": "", "phone": "", "hours": "",
              "rating": "", "category": ""}

    # Extract business name from URL path (works for all GBP URL formats)
    result["businessName"] = _name_from_gbp_url(gbp_url)

    # Try direct HTML fetch — Google embeds some OGP/meta tags even without JS
    text = ""
    try:
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS, follow_redirects=True, timeout=12
        ) as client:
            resp = await client.get(gbp_url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                # Try meta description and og:description
                for attr in ("og:description", "description"):
                    meta = soup.find("meta", attrs={"property": attr}) or \
                           soup.find("meta", attrs={"name": attr})
                    if meta and meta.get("content"):
                        text += meta["content"] + " "
                # Try page title
                if soup.title and soup.title.string:
                    text += soup.title.string + " "
    except Exception as e:
        print(f"GBP direct fetch error: {e}")

    if not text.strip():
        return result  # return with name only

    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Extract business information from this Google Maps/Business Profile page text. "
                                    "Return ONLY a JSON object with these keys: "
                                    "businessName, address, phone, hours, rating, category. "
                                    "Use empty string for fields not found. No markdown, just JSON."
                                ),
                            },
                            {"role": "user", "content": f"Extract:\n\n{text[:5000]}"},
                        ],
                        "max_tokens": 400,
                        "temperature": 0.1,
                    },
                )
            data = resp.json()
            parsed = _parse_llm_json(data["choices"][0]["message"]["content"].strip())
            for key in result:
                if parsed.get(key):
                    result[key] = str(parsed[key])[:300]
        except Exception as e:
            print(f"GBP Groq extract error: {e}")

    if not result["businessName"]:
        result["businessName"] = _name_from_gbp_url(gbp_url)

    print(f"GBP extracted fields: {[k for k, v in result.items() if v]}")
    return result


def _parse_llm_json(text: str) -> dict:
    """Robustly parse JSON from LLM output that may have markdown code fences."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # Find outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _name_from_gbp_url(url: str) -> str:
    """Extract readable business name from a Google Maps URL path."""
    m = re.search(r"/place/([^/@?&#]+)", url)
    if m:
        raw = m.group(1)
        return raw.replace("+", " ").replace("%20", " ").replace("%2C", ",").strip()
    return ""


async def _groq_extract(text: str, include_address: bool = False) -> dict:
    """Use llama-3.3-70b-versatile to extract business fields from raw text."""
    keys = "about, services, pricing" + (", address, phone" if include_address else "")
    extracted = {"about": "", "services": "", "pricing": "", "address": "", "phone": ""}
    if not GROQ_API_KEY or not text.strip():
        return extracted
    trimmed = text[:5000]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract business information from website/page text. "
                                f"Return ONLY a JSON object with these keys: {keys}. "
                                "For 'about': 2-3 sentences about what the business does. "
                                "For 'services': comma-separated list of services or products. "
                                "For 'pricing': all price info found (packages, rates, tariffs). "
                                "For 'address': full physical address if found. "
                                "Use empty string for fields not found. No markdown, just JSON."
                            ),
                        },
                        {"role": "user", "content": f"Extract business info:\n\n{trimmed}"},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.1,
                },
            )
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = _parse_llm_json(content)
        for key in extracted:
            if parsed.get(key):
                extracted[key] = str(parsed[key])[:600]
        print(f"Groq extract keys found: {[k for k, v in extracted.items() if v]}")
    except Exception as e:
        print(f"Groq extract error: {e}")
    return extracted


# Keep alias for compatibility
_extract_with_groq = _groq_extract


async def _scrape_website(website_url: str) -> dict:
    """Extract business info from a website via Jina Reader (+ httpx fallback) + Groq."""
    result = {"about": "", "services": "", "pricing": "", "address": "", "contact": "", "faqs": ""}

    base = website_url.rstrip("/")
    # Fetch 3 most valuable pages concurrently (homepage always, plus contact + about)
    pages_to_fetch = [
        website_url,
        f"{base}/contact",
        f"{base}/about",
    ]

    raw_texts = await asyncio.gather(*[_jina_read(u) for u in pages_to_fetch], return_exceptions=True)
    pages = [t for t in raw_texts if isinstance(t, str) and len(t.strip()) > 100]

    if not pages:
        print(f"Website scrape: no content retrieved for {website_url}")
        return result

    combined = "\n\n---\n\n".join(pages)[:8000]

    # Single Groq call to extract all fields
    groq_data = await _groq_extract(combined, include_address=True)
    result["about"]    = groq_data.get("about", "")
    result["services"] = groq_data.get("services", "")
    result["pricing"]  = groq_data.get("pricing", "")
    result["address"]  = groq_data.get("address", "")

    # Regex for phone/email (reliable in clean text)
    phones = re.findall(r"(\+?\d[\d\s\-().]{7,}\d)", combined)
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", combined)
    parts = []
    if phones:
        parts.append("Ph: " + ", ".join(dict.fromkeys(phones[:3])))
    if emails:
        parts.append("Email: " + ", ".join(dict.fromkeys(emails[:3])))
    result["contact"] = "  ".join(parts)

    print(f"Website extracted fields: {[k for k, v in result.items() if v]}")
    return result


@app.post("/fetch-business-context")
async def fetch_business_context_endpoint(body: FetchBusinessContextRequest):
    gbp_data: dict = {}
    web_data: dict = {}

    # Run GBP and website scrapes concurrently
    tasks = []
    if body.gbp_url:
        tasks.append(("gbp", asyncio.create_task(_scrape_gbp(body.gbp_url))))
    if body.website_url:
        tasks.append(("web", asyncio.create_task(_scrape_website(body.website_url))))

    for key, task in tasks:
        try:
            data = await task
            if key == "gbp":
                gbp_data = data
            else:
                web_data = data
        except Exception as e:
            print(f"{key} scrape task error: {e}")

    result = {
        "businessName":  gbp_data.get("businessName") or "",
        "address":       gbp_data.get("address") or web_data.get("address") or "",
        "phone":         gbp_data.get("phone") or web_data.get("contact") or "",
        "hours":         gbp_data.get("hours") or "",
        "rating":        gbp_data.get("rating") or "",
        "category":      gbp_data.get("category") or "",
        "about":         web_data.get("about") or "",
        "services":      web_data.get("services") or "",
        "pricing":       web_data.get("pricing") or "",
        "contact":       web_data.get("contact") or "",
        "faqs":          web_data.get("faqs") or "",
        "location":      "",
        "instagram":     "",
        "facebook":      "",
        "otaPlatforms":  "",
        "activities":    "",
        "paidActivities":"",
        "accommodations":"",
        "events":        "",
        "fetched":       True,
    }

    # Persist to Firestore — skip empty strings so existing data is never overwritten
    try:
        db = get_db()
        to_save = {k: v for k, v in result.items() if v != "" and v is not None}
        to_save["fetched"] = True  # always mark as fetched
        db.collection("business_context").document(body.user_id).set(
            to_save, merge=True
        )
        print(f"Saved business context for user {body.user_id}: {list(to_save.keys())}")
    except Exception as e:
        print(f"Firestore save error: {e}")

    return result


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
                                    "timestamp": datetime.utcnow().strftime("%I:%M %p"),
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
    finally:
        if session_doc_ref and session_id:
            try:
                db = get_db()
                db.collection("call_sessions").document(session_id).update({
                    "status": "completed",
                    "endTime": SERVER_TIMESTAMP,
                    "totalExchanges": len(exchanges)
                })
                print(f"Session {session_id} completed with {len(exchanges)} exchanges")
            except Exception as fe:
                print(f"Session finalize error: {fe}")

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