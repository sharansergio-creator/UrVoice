from fastapi import FastAPI, Request, WebSocket, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import os
from google import genai
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
from firebase_admin import credentials, firestore, messaging
from google.cloud.firestore import SERVER_TIMESTAMP, ArrayUnion, Increment, DELETE_FIELD
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize Firebase Admin SDK
_firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS")
if _firebase_creds_json and not firebase_admin._apps:
    _cred = credentials.Certificate(json.loads(_firebase_creds_json))
    firebase_admin.initialize_app(_cred)

def get_db():
    return firestore.client()

def create_vad(aggressiveness: int = 2):
    """
    Create a WebRTC VAD instance.
    aggressiveness: 0 (least aggressive) to 3 (most aggressive)
    Higher = more aggressive filtering of non-speech
    2 is recommended for phone calls
    """
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    return vad

def is_speech(vad, pcm_chunk: bytes, sample_rate: int = 8000) -> bool:
    """
    Check if a PCM audio chunk contains speech.
    WebRTC VAD requires chunks of exactly 10ms, 20ms, or 30ms.
    At 8000Hz, 16-bit mono:
    - 10ms = 160 bytes
    - 20ms = 320 bytes  
    - 30ms = 480 bytes
    We use 20ms frames (320 bytes).
    """
    import webrtcvad
    FRAME_SIZE = 320  # 20ms at 8000Hz 16-bit mono
    
    if len(pcm_chunk) < FRAME_SIZE:
        return False
    
    # Check multiple 20ms frames and return True if any contains speech
    speech_frames = 0
    total_frames = 0
    
    for i in range(0, len(pcm_chunk) - FRAME_SIZE + 1, FRAME_SIZE):
        frame = pcm_chunk[i:i + FRAME_SIZE]
        if len(frame) == FRAME_SIZE:
            try:
                if vad.is_speech(frame, sample_rate):
                    speech_frames += 1
                total_frames += 1
            except Exception:
                continue
    
    if total_frames == 0:
        return False
    
    # Consider as speech if more than 50% of frames contain speech
    return (speech_frames / total_frames) > 0.5

async def get_user_id_from_phone(called_number: str) -> str:
    """Look up which userId owns this Twilio number from Firestore phone_mappings."""
    try:
        db = get_db()
        # Normalize number - try with and without +
        numbers_to_try = [called_number]
        if called_number.startswith("+"):
            numbers_to_try.append(called_number.replace("+", ""))
        for number in numbers_to_try:
            doc = db.collection("phone_mappings").document(number).get()
            if doc.exists:
                user_id = doc.to_dict().get("userId")
                if user_id:
                    print(f"Resolved userId {user_id} for number {number}")
                    return user_id
    except Exception as e:
        print(f"get_user_id_from_phone error: {e}")
    # Fallback to default user if no mapping found
    print(f"No phone_mapping found for {called_number}, using fallback")
    return "MmBTqzNf5OgIOIctQKPiRQezadi1"

async def append_exchange_to_session(doc_ref, exchange: dict):
    """Append one exchange dict to the session document's exchanges array."""
    try:
        doc_ref.update({"exchanges": ArrayUnion([exchange])})
    except Exception as e:
        print(f"Session exchange append error: {e}")

async def send_fcm_notification(user_id: str, title: str, body: str, data: dict):
    try:
        db = get_db()
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            print(f"No user doc found for {user_id}")
            return
        fcm_token = user_doc.to_dict().get("fcmToken")
        if not fcm_token:
            print(f"No FCM token for user {user_id}")
            return
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in data.items()},
            token=fcm_token,
            android=messaging.AndroidConfig(priority="high")
        )
        response = messaging.send(message)
        print(f"FCM notification sent: {response}")
    except Exception as e:
        print(f"FCM error: {e}")


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
    user_id: str = "MmBTqzNf5OgIOIctQKPiRQezadi1"


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

    if GEMINI_API_KEY:
        try:
            gbp_client = genai.Client(api_key=GEMINI_API_KEY)
            gbp_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gbp_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[{"role": "user", "parts": [{"text": f"Extract:\n\n{text[:5000]}"}]}],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=(
                            "Extract business information from this Google Maps/Business Profile page text. "
                            "Return ONLY a JSON object with these keys: "
                            "businessName, address, phone, hours, rating, category. "
                            "Use empty string for fields not found. No markdown, just JSON."
                        ),
                        max_output_tokens=400,
                    )
                )
            )
            parsed = _parse_llm_json(gbp_response.text.strip())
            for key in result:
                if parsed.get(key):
                    result[key] = str(parsed[key])[:300]
        except Exception as e:
            print(f"GBP Gemini extract error: {e}")

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


async def _gemini_extract(text: str, include_address: bool = False) -> dict:
    """Use gemini-2.5-flash to extract business fields from raw text."""
    keys = "about, services, pricing" + (", address, phone" if include_address else "")
    extracted = {"about": "", "services": "", "pricing": "", "address": "", "phone": ""}
    if not GEMINI_API_KEY or not text.strip():
        return extracted
    trimmed = text[:5000]
    try:
        extract_client = genai.Client(api_key=GEMINI_API_KEY)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: extract_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": f"Extract business info:\n\n{trimmed}"}]}],
                config=genai.types.GenerateContentConfig(
                    system_instruction=(
                        "You extract business information from website/page text. "
                        f"Return ONLY a JSON object with these keys: {keys}. "
                        "For 'about': 2-3 sentences about what the business does. "
                        "For 'services': comma-separated list of services or products. "
                        "For 'pricing': all price info found (packages, rates, tariffs). "
                        "For 'address': full physical address if found. "
                        "Use empty string for fields not found. No markdown, just JSON."
                    ),
                    max_output_tokens=600,
                )
            )
        )
        parsed = _parse_llm_json(response.text.strip())
        for key in extracted:
            if parsed.get(key):
                extracted[key] = str(parsed[key])[:600]
        print(f"Gemini extract keys found: {[k for k, v in extracted.items() if v]}")
    except Exception as e:
        print(f"Gemini extract error: {e}")
    return extracted





async def _scrape_website(website_url: str) -> dict:
    """Extract business info from a website via Jina Reader (+ httpx fallback) + Gemini."""
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

    # Single Gemini call to extract all fields
    gemini_data = await _gemini_extract(combined, include_address=True)
    result["about"]    = gemini_data.get("about", "")
    result["services"] = gemini_data.get("services", "")
    result["pricing"]  = gemini_data.get("pricing", "")
    result["address"]  = gemini_data.get("address", "")

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


class ProvisionNumberRequest(BaseModel):
    user_id: str
    country_code: str = "US"

@app.post("/provision-number")
async def provision_number(body: ProvisionNumberRequest):
    """
    Buy a new Twilio phone number, set its webhook to this backend,
    and register the mapping in Firestore phone_mappings.
    """
    try:
        # Search for available numbers
        search_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/AvailablePhoneNumbers/{body.country_code}/Local.json"
        async with httpx.AsyncClient() as client:
            search_resp = await client.get(
                search_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                params={"VoiceEnabled": "true", "Limit": 1}
            )
            if search_resp.status_code != 200:
                return {"error": f"Number search failed: {search_resp.text}"}

            numbers = search_resp.json().get("available_phone_numbers", [])
            if not numbers:
                return {"error": "No available numbers found"}

            phone_number = numbers[0]["phone_number"]
            print(f"Found available number: {phone_number}")

            # Purchase the number and set webhook
            purchase_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json"
            purchase_resp = await client.post(
                purchase_url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                data={
                    "PhoneNumber": phone_number,
                    "VoiceUrl": "https://urvoice-production.up.railway.app/incoming-call",
                    "VoiceMethod": "POST",
                    "StatusCallback": "https://urvoice-production.up.railway.app/call-status",
                    "StatusCallbackMethod": "POST",
                }
            )
            if purchase_resp.status_code not in (200, 201):
                return {"error": f"Number purchase failed: {purchase_resp.text}"}

            purchased = purchase_resp.json()
            assigned_number = purchased.get("phone_number")
            print(f"Purchased number: {assigned_number}")

            # Save to Firestore phone_mappings
            db = get_db()
            db.collection("phone_mappings").document(assigned_number).set({
                "userId": body.user_id,
                "assignedAt": SERVER_TIMESTAMP,
                "country": body.country_code
            })
            print(f"Saved phone_mapping: {assigned_number} -> {body.user_id}")

            # Also save the number to users/{userId}
            db.collection("users").document(body.user_id).update({
                "twilioNumber": assigned_number
            })

            return {
                "success": True,
                "phoneNumber": assigned_number,
                "userId": body.user_id
            }

    except Exception as e:
        print(f"provision_number error: {e}")
        return {"error": str(e)}


async def get_elevenlabs_voice_id(user_id: str, language: str = "en") -> str | None:
    """
    Get ElevenLabs voice ID for a specific language.
    Checks voiceClones map first (new multi-language), 
    falls back to elevenLabsVoiceId (legacy single voice).
    """
    try:
        db = get_db()
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            # New multi-language voice clones map
            voice_clones = data.get("voiceClones", {})
            if voice_clones and language in voice_clones:
                return voice_clones[language]
            # Legacy fallback — single voice ID
            legacy = data.get("elevenLabsVoiceId")
            if legacy:
                return legacy
    except Exception as e:
        print(f"get_elevenlabs_voice_id error: {e}")
    return None


@app.post("/clone-voice")
async def clone_voice(
    user_id: str = Form(...),
    language: str = Form(default="en"),
    audio: UploadFile = File(...)
):
    """
    Receive audio sample, send to ElevenLabs to create voice clone,
    save voice_id to Firestore users/{userId}/voiceClones/{language}.
    """
    try:
        audio_bytes = await audio.read()
        print(f"Received audio for cloning: {len(audio_bytes)} bytes for user {user_id}, language: {language}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files={"files": (audio.filename or f"voice_{language}.m4a", audio_bytes, "audio/m4a")},
                data={
                    "name": f"UrVoice_{user_id[:8]}_{language}",
                    "description": f"Voice clone for UrVoice AI assistant - {language}"
                },
                timeout=60
            )

            if response.status_code != 200:
                print(f"ElevenLabs clone error: {response.status_code} {response.text}")
                return {"error": f"Voice cloning failed: {response.text}"}

            result = response.json()
            voice_id = result.get("voice_id")
            print(f"Voice cloned successfully: {voice_id} for user {user_id}, language: {language}")

            # Save to voiceClones map in Firestore
            db = get_db()
            db.collection("users").document(user_id).set(
                {"voiceClones": {language: voice_id}},
                merge=True
            )

            # Also update legacy elevenLabsVoiceId for English (backward compat)
            if language == "en":
                db.collection("users").document(user_id).update({
                    "elevenLabsVoiceId": voice_id
                })

            return {"success": True, "voiceId": voice_id, "language": language}

    except Exception as e:
        print(f"clone_voice error: {e}")
        return {"error": str(e)}


@app.post("/delete-voice")
async def delete_voice(request: Request):
    """Remove ElevenLabs voice clone for a specific language and clear from Firestore."""
    try:
        body = await request.json()
        user_id = body.get("user_id")
        voice_id = body.get("voice_id")
        language = body.get("language", "en")

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                timeout=30
            )
            print(f"ElevenLabs delete response: {response.status_code}")

        # Remove from voiceClones map
        db = get_db()
        db.collection("users").document(user_id).update({
            f"voiceClones.{language}": DELETE_FIELD
        })

        # Clear legacy field if English
        if language == "en":
            db.collection("users").document(user_id).update({
                "elevenLabsVoiceId": None
            })

        return {"success": True, "language": language}
    except Exception as e:
        print(f"delete_voice error: {e}")
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "UrVoice backend running"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    called_number = form_data.get("To", "unknown")
    call_sid = form_data.get("CallSid", "unknown")
    host = request.headers.get("host")
    user_id = await get_user_id_from_phone(called_number)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/audio-stream">
            <Parameter name="CallSid" value="{call_sid}"/>
            <Parameter name="From" value="{caller_number}"/>
            <Parameter name="UserId" value="{user_id}"/>
        </Stream>
    </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request):
    form_data = await request.form()
    caller_number = form_data.get("From", "unknown")
    called_number = form_data.get("To", "unknown")
    call_sid = form_data.get("CallSid", "unknown")
    call_status = form_data.get("CallStatus", "unknown")
    print(f"Call status: {call_status}, from: {caller_number}")
    business_user_id = await get_user_id_from_phone(called_number)
    caller_info = await get_caller_info(business_user_id, caller_number)
    caller_name = caller_info.get("name") or caller_number
    caller_type = caller_info.get("type", "UNKNOWN")
    if call_status == "ringing":
        asyncio.create_task(send_fcm_notification(
            business_user_id,
            "📞 Incoming Call",
            f"{caller_name} is calling",
            {
                "callSid": call_sid,
                "callerNumber": caller_number,
                "callerName": caller_name,
                "callerType": caller_type,
                "type": "CALL_INCOMING"
            }
        ))
    return PlainTextResponse("", status_code=200)

async def get_caller_info(user_id: str, caller_number: str) -> dict:
    """Retrieve caller info from Firestore contact_permissions/{user_id}/contacts/{caller_number}."""
    try:
        db = get_db()
        doc = (
            db.collection("contact_permissions")
            .document(user_id)
            .collection("contacts")
            .document(caller_number)
            .get()
        )
        if doc.exists:
            data = doc.to_dict()
            return {
                "name": data.get("name"),
                "type": data.get("type", "UNKNOWN"),
                "totalCalls": data.get("totalCalls", 0),
            }
    except Exception as e:
        print(f"get_caller_info error: {e}")
    return {"name": None, "type": "UNKNOWN", "totalCalls": 0}


async def save_caller_info(user_id: str, caller_number: str, name: str, call_type: str):
    """Save or update caller info in Firestore contact_permissions/{user_id}/contacts/{caller_number}."""
    try:
        db = get_db()
        ref = (
            db.collection("contact_permissions")
            .document(user_id)
            .collection("contacts")
            .document(caller_number)
        )
        doc = ref.get()
        if doc.exists:
            ref.update({
                "name": name,
                "type": call_type,
                "lastCall": SERVER_TIMESTAMP,
                "totalCalls": Increment(1),
            })
        else:
            ref.set({
                "name": name,
                "type": call_type,
                "firstCall": SERVER_TIMESTAMP,
                "lastCall": SERVER_TIMESTAMP,
                "totalCalls": 1,
            })
        print(f"Saved caller info: {caller_number} -> {name} ({call_type})")
    except Exception as e:
        print(f"save_caller_info error: {e}")


async def update_business_hours_check(user_id: str) -> bool:
    """Return True if current time (IST) falls within any enabled business hour slot, False if closed."""
    try:
        db = get_db()
        doc = db.collection("business_context").document(user_id).get()
        if not doc.exists:
            return True
        data = doc.to_dict()
        slots = data.get("businessHours", [])
        if not slots:
            return True
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        day_name = now.strftime("%A")
        current_minutes = now.hour * 60 + now.minute
        for slot in slots:
            if not slot.get("enabled", False):
                continue
            if slot.get("day", "").lower() != day_name.lower():
                continue
            open_time = slot.get("openTime", "00:00")
            close_time = slot.get("closeTime", "23:59")
            try:
                open_h, open_m = map(int, open_time.split(":"))
                close_h, close_m = map(int, close_time.split(":"))
                if open_h * 60 + open_m <= current_minutes <= close_h * 60 + close_m:
                    return True
            except Exception:
                continue
        return False
    except Exception as e:
        print(f"update_business_hours_check error: {e}")
        return True


def _format_business_hours(slots: list) -> str:
    """Format businessHours array into a human-readable string."""
    parts = []
    for slot in slots:
        if not slot.get("enabled", False):
            continue
        day = slot.get("day", "")
        open_time = slot.get("openTime", "")
        close_time = slot.get("closeTime", "")
        if day and open_time and close_time:
            parts.append(f"{day} {open_time} to {close_time}")
    return ", ".join(parts) if parts else "our regular business hours"


async def get_call_settings(user_id: str) -> dict:
    """Read call handling settings from Firestore call_settings/{userId}."""
    try:
        db = get_db()
        doc = db.collection("call_settings").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        print(f"get_call_settings error: {e}")
    return {"answerMode": "ALWAYS"}


@app.websocket("/audio-stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    audio_chunks = []
    stream_sid = None
    caller_number = None
    business_context = ""
    speaking = False
    silence_frames = 0
    last_response_time = 0.0  # timestamp of last AI response completion
    is_playing = False
    conversation_history = []
    session_id = None
    session_doc_ref = None
    exchanges = []
    caller_name = None
    caller_type = "UNKNOWN"
    name_attempts = 0
    is_blocked = False
    is_after_hours = False
    hours_string = ""
    name_collected = False
    SILENCE_LIMIT = 30
    # Initialize WebRTC VAD
    import webrtcvad as _webrtcvad
    vad = _webrtcvad.Vad(3)  # aggressiveness 3 = most aggressive filtering

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                # Send keep-alive to prevent Twilio from closing WebSocket
                try:
                    await websocket.send_text(json.dumps({"event": "ping"}))
                except Exception:
                    break
                continue
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                caller_number = (
                    data["start"].get("customParameters", {}).get("From") or
                    data["start"].get("customParameters", {}).get("from") or
                    data["start"].get("customParameters", {}).get("CallFrom") or
                    data["start"].get("from") or
                    data["start"].get("From") or
                    "unknown"
                )
                business_user_id = (
                    data["start"].get("customParameters", {}).get("UserId") or
                    "MmBTqzNf5OgIOIctQKPiRQezadi1"
                )
                print(f"Business userId for this call: {business_user_id}")
                print(f"Stream start data: {json.dumps(data['start'], indent=2)}")
                print(f"Stream started: {stream_sid}, caller: {caller_number}")
                # Fetch business context once per call
                business_context = await fetch_business_context(business_user_id)
                print(f"Business context loaded: {bool(business_context)}")

                call_settings = await get_call_settings(business_user_id)
                answer_mode = call_settings.get("answerMode", "ALWAYS")
                print(f"Answer mode: {answer_mode}")

                if answer_mode == "NEVER":
                    sorry = "Sorry, we are not available to take calls right now. Please try again later."
                    audio_chunks.clear()
                    speaking = False
                    silence_frames = 0
                    await send_audio_response(websocket, stream_sid, sorry, business_user_id)
                    return

                # Extract business name from context for use in greetings
                biz_name = "our business"
                if business_context:
                    first_line = business_context.splitlines()[0]
                    raw_name = first_line.replace("You are the AI phone assistant for ", "")
                    comma_idx = raw_name.find(",")
                    biz_name = (raw_name[:comma_idx] if comma_idx != -1 else raw_name).strip().rstrip(".")

                # Identify caller from contact_permissions
                caller_info = await get_caller_info(business_user_id, caller_number)
                caller_name = caller_info["name"]
                caller_type = caller_info["type"]
                name_collected = caller_name is not None
                is_blocked = caller_type == "BLOCKED"

                # After-hours check
                is_open = await update_business_hours_check(business_user_id)
                if not is_open and not is_blocked:
                    is_after_hours = True
                    try:
                        db_h = get_db()
                        doc_h = db_h.collection("business_context").document(business_user_id).get()
                        if doc_h.exists:
                            hours_string = _format_business_hours(
                                doc_h.to_dict().get("businessHours", [])
                            )
                    except Exception as e:
                        print(f"After-hours hours fetch error: {e}")
                    if not hours_string:
                        hours_string = "our regular business hours"
                    print(f"After hours call from {caller_number}")

                # Select greeting based on caller type / after-hours
                if caller_type == "BLOCKED":
                    greeting = "I'm sorry, this number is not able to reach us. Goodbye."
                elif is_after_hours:
                    greeting = (
                        f"Thank you for calling {biz_name}. We are currently closed. "
                        f"Our business hours are {hours_string}. Please call back during our working hours, "
                        f"or leave your name and number and we will call you back."
                    )
                    business_context = (
                        f"You are the AI phone assistant for {biz_name}. "
                        f"The business is currently closed. Your ONLY job is to: "
                        f"1. Apologize for being unavailable. "
                        f"2. Tell the caller the business hours: {hours_string}. "
                        f"3. Ask for their name and callback number. "
                        f"4. Thank them and end the call politely. "
                        f"Do NOT discuss bookings, pricing, or services in detail."
                    )
                elif caller_type == "VIP":
                    greeting = f"Hello! Thank you for calling {biz_name}, please hold while we connect you."
                    # TODO: Send FCM notification to owner
                elif caller_type == "CUSTOMER" and caller_name:
                    greeting = f"Welcome back {caller_name}! Thank you for calling {biz_name}, how can I help you today?"
                else:
                    greeting = f"Thank you for calling {biz_name}, may I know who is calling please?"

                # Create session document in Firestore
                session_id = str(uuid.uuid4())
                db = get_db()
                session_doc_ref = db.collection("call_sessions").document(session_id)
                session_doc_ref.set({
                    "sessionId": session_id,
                    "userId": business_user_id,
                    "callerNumber": caller_number or "unknown",
                    "callerName": caller_name,
                    "category": "AFTER_HOURS" if is_after_hours else caller_type,
                    "startTime": SERVER_TIMESTAMP,
                    "status": "active",
                    "exchanges": [],
                })
                print(f"Session created: {session_id}")

                asyncio.create_task(send_fcm_notification(
                    business_user_id,
                    "📞 Incoming Call",
                    f"{caller_name or 'Unknown Caller'} is calling",
                    {"sessionId": session_id, "callerName": caller_name or "", "type": "CALL_STARTED"}
                ))

                audio_chunks.clear()
                speaking = False
                silence_frames = 0
                is_playing = True
                await send_audio_response(websocket, stream_sid, greeting, business_user_id)
                is_playing = False
                audio_chunks.clear()
                speaking = False
                silence_frames = 0
                if is_blocked:
                    break

            elif data["event"] == "media":
                if is_playing:
                    audio_chunks.clear()
                    speaking = False
                    silence_frames = 0
                    continue

                raw_chunk = base64.b64decode(data["media"]["payload"])
                audio_chunks.append(data["media"]["payload"])

                pcm_chunk = mulaw_chunk_to_pcm(raw_chunk)
                
                # WebRTC VAD - more accurate than RMS threshold
                has_speech = is_speech(vad, pcm_chunk)
                
                if has_speech:
                    speaking = True
                    silence_frames = 0
                elif speaking:
                    silence_frames += 1
                    if silence_frames >= SILENCE_LIMIT:
                        speaking = False
                        silence_frames = 0

                        import time
                        # Ignore speech detected within 1.5 seconds of last AI response
                        if time.time() - last_response_time < 1.5:
                            audio_chunks.clear()
                            speaking = False
                            silence_frames = 0
                            continue

                        chunks_to_process = audio_chunks.copy()
                        audio_chunks.clear()

                        raw_mulaw = b"".join(base64.b64decode(c) for c in chunks_to_process)
                        wav_bytes = mulaw_to_wav(raw_mulaw)
                        # Use language from last AI response as hint for next STT call
                        last_lang = exchanges[-1]["language"] if exchanges else "en-IN"
                        # Skip very short audio - likely silence or noise
                        duration_seconds = len(raw_mulaw) / 8000
                        if duration_seconds < 0.8:
                            print(f"Audio too short ({duration_seconds:.2f}s), skipping STT")
                            audio_chunks.clear()
                            continue

                        transcript = await transcribe(wav_bytes, language_hint=last_lang)
                        print(f"Caller said: {transcript}")

                        # Filter out STT hallucinations - gibberish short responses
                        transcript_clean = transcript.strip() if transcript else ""
                        word_count = len(transcript_clean.split())
                        is_likely_hallucination = (
                            word_count == 1 and len(transcript_clean) <= 4
                        )
                        # Filter pure noise descriptions like (static), (silence)
                        # But keep mixed ones like "(silence) Nope." → extract real part
                        import re as _re
                        # Remove parenthetical noise descriptions from transcript
                        cleaned_transcript = _re.sub(r'\([^)]*\)', '', transcript_clean).strip()
                        cleaned_transcript = _re.sub(r'\s+', ' ', cleaned_transcript).strip()
                        is_noise_description = len(cleaned_transcript) < 2
                        # Use cleaned version if it has content
                        if cleaned_transcript and len(cleaned_transcript) >= 2:
                            transcript_clean = cleaned_transcript
                        if transcript_clean and len(transcript_clean) > 2 and not is_likely_hallucination and not is_noise_description:
                            conversation_history.append({"role": "user", "content": transcript})

                            # Augment system context with name-collection directive for UNKNOWN callers
                            effective_context = business_context
                            if not is_after_hours and not name_collected and name_attempts < 2:
                                effective_context = business_context + (
                                    "\n\nThe caller has not given their name yet. Your ONLY job right now "
                                    "is to get their name. Ask: 'May I know your name please?' "
                                    "If they give a name, start your response with 'NAME:[their name]' "
                                    "on its own line, then continue normally on the next line. "
                                    "If they don't give a clear name, ask once more politely."
                                )

                            detected_input_lang = detect_language(transcript)
                            lang_instruction = f"\n\nCRITICAL: The caller just spoke in {'Kannada' if detected_input_lang == 'kn-IN' else 'English' if detected_input_lang == 'en-IN' else detected_input_lang}. You MUST respond in that exact same language now. Do not continue in the previous language."
                            ai_response = await get_ai_response(conversation_history, effective_context + lang_instruction)
                            print(f"AI response: {ai_response}")

                            # Name extraction for UNKNOWN callers
                            if not name_collected and name_attempts < 2 and ai_response:
                                if ai_response.startswith("NAME:"):
                                    rest = ai_response[5:]
                                    newline_pos = rest.find("\n")
                                    if newline_pos != -1:
                                        extracted_name = rest[:newline_pos].strip()
                                        ai_response = rest[newline_pos + 1:].strip()
                                    else:
                                        name_match = re.match(
                                            r"([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(.*)", rest, re.DOTALL
                                        )
                                        if name_match:
                                            extracted_name = name_match.group(1).strip()
                                            ai_response = name_match.group(2).strip()
                                        else:
                                            extracted_name = rest.strip()
                                            ai_response = ""
                                    if extracted_name:
                                        caller_name = extracted_name
                                        name_collected = True
                                        caller_type = "CUSTOMER"
                                        asyncio.create_task(
                                            save_caller_info(business_user_id, caller_number, caller_name, "CUSTOMER")
                                        )
                                        if session_doc_ref:
                                            try:
                                                session_doc_ref.update({
                                                    "callerName": caller_name,
                                                    "category": "CUSTOMER",
                                                })
                                            except Exception as e:
                                                print(f"Session name update error: {e}")
                                        print(f"Caller name identified: {caller_name}")
                                else:
                                    name_attempts += 1
                                    if name_attempts >= 2:
                                        asyncio.create_task(
                                            save_caller_info(business_user_id, caller_number, "Unknown", "SPAM")
                                        )
                                        if session_doc_ref:
                                            try:
                                                session_doc_ref.update({"category": "SPAM"})
                                            except Exception as e:
                                                print(f"Session spam update error: {e}")
                                        print(f"Caller {caller_number} logged as SPAM after {name_attempts} name attempts")
                                        sorry_msg = "I'm sorry I couldn't get your name, please call back when ready. Goodbye."
                                        audio_chunks.clear()
                                        speaking = False
                                        silence_frames = 0
                                        is_playing = True
                                        await send_audio_response(websocket, stream_sid, sorry_msg, business_user_id)
                                        is_playing = False
                                        audio_chunks.clear()
                                        speaking = False
                                        silence_frames = 0
                                        break

                            if ai_response and stream_sid:
                                conversation_history.append({"role": "assistant", "content": ai_response})

                                # Detect language from AI response (more reliable than transcript)
                                # because STT returns English text even for Indian language speech
                                detected_lang = detect_language(ai_response)
                                if detected_lang == "en-IN":
                                    # Double check transcript in case AI responded in English
                                    # but caller was speaking Indian language
                                    transcript_lang = detect_language(transcript)
                                    detected_lang = transcript_lang if transcript_lang != "en-IN" else "en-IN"

                                audio_chunks.clear()
                                speaking = False
                                silence_frames = 0
                                is_playing = True
                                await send_audio_response(websocket, stream_sid, ai_response, business_user_id)
                                is_playing = False
                                audio_chunks.clear()
                                speaking = False
                                silence_frames = 0
                                import time
                                last_response_time = time.time()

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
                        update_data = {
                            "status": "completed",
                            "endTime": SERVER_TIMESTAMP,
                            "totalExchanges": len(exchanges),
                        }
                        if is_after_hours:
                            update_data["category"] = "AFTER_HOURS"
                        session_doc_ref.update(update_data)
                        print(f"Session {session_id} completed with {len(exchanges)} exchange(s)")
                    except Exception as e:
                        print(f"Session close error: {e}")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if session_doc_ref and session_id:
            try:
                update_data = {
                    "status": "completed",
                    "endTime": SERVER_TIMESTAMP,
                    "totalExchanges": len(exchanges)
                }
                if is_after_hours:
                    update_data["category"] = "AFTER_HOURS"
                db = get_db()
                db.collection("call_sessions").document(session_id).update(update_data)
                print(f"Session {session_id} completed with {len(exchanges)} exchanges")
                asyncio.create_task(send_fcm_notification(
                    business_user_id,
                    "📋 Call Completed",
                    f"Call ended - {len(exchanges)} exchanges",
                    {"sessionId": session_id, "type": "CALL_ENDED"}
                ))
            except Exception as fe:
                print(f"Session finalize error: {fe}")

async def send_audio_response(websocket: WebSocket, stream_sid: str, text: str, user_id: str = None):
    try:
        import time
        total_start = time.time()
        audio_bytes = await text_to_speech(text, user_id)
        if not audio_bytes:
            return

        # text_to_speech returns the correct format:
        # - ElevenLabs pcm_22050: raw 16-bit PCM at 22050Hz (needs resample + mulaw)
        # - Sarvam: raw 16-bit PCM at 8000Hz (needs mulaw conversion only)
        # We detect which by checking if user_id resulted in ElevenLabs usage
        language = detect_language(text)
        # Use ElevenLabs decode for ANY language when user has a voice clone
        # because text_to_speech() returns ElevenLabs MP3 for all languages when clone exists
        use_elevenlabs = (user_id is not None and

        if use_elevenlabs:
            try:
                import miniaudio as _miniaudio
                import audioop
                # Strip ID3 tag if present before decoding
                audio_to_decode = audio_bytes
                if audio_bytes[:3] == b'ID3':
                    # Find end of ID3 tag
                    id3_size = ((audio_bytes[6] & 0x7f) << 21 |
                               (audio_bytes[7] & 0x7f) << 14 |
                               (audio_bytes[8] & 0x7f) << 7 |
                               (audio_bytes[9] & 0x7f)) + 10
                    audio_to_decode = audio_bytes[id3_size:]
                    print(f"Stripped ID3 tag: {id3_size} bytes, remaining: {len(audio_to_decode)}")

                decoded = _miniaudio.decode(
                    audio_to_decode,
                    output_format=_miniaudio.SampleFormat.SIGNED16,
                    nchannels=1,
                    sample_rate=8000
                )
                pcm_8k = bytes(decoded.samples)
                if len(pcm_8k) % 2 != 0:
                    pcm_8k = pcm_8k[:-1]
                mulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
                print(f"ElevenLabs MP3 decoded: {len(audio_bytes)} -> {len(pcm_8k)} -> {len(mulaw_audio)}")
            except Exception as e:
                print(f"ElevenLabs decode error: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to Sarvam for this response
                fallback = await sarvam_tts(text, detect_language(text))
                import audioop
                mulaw_audio = audioop.lin2ulaw(fallback, 2) if fallback else b""
        else:
            import audioop
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            mulaw_audio = audioop.lin2ulaw(audio_bytes, 2)

        payload = base64.b64encode(mulaw_audio).decode("utf-8")
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        await websocket.send_text(json.dumps(message))
        total_latency = round((time.time() - total_start) * 1000)
        print(f"[LATENCY] TOTAL response: {total_latency}ms | text: '{text[:40]}'")
        print(f"Sent audio response for: {text[:50]}")

        word_count = len(text.split())
        wait_time = max(1.5, word_count * 0.35)
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

async def text_to_speech(text: str, user_id: str = None) -> bytes:
    language = detect_language(text)

    if user_id and ELEVENLABS_API_KEY:
        if language == "en-IN":
            # For English use the English clone
            voice_id = await get_elevenlabs_voice_id(user_id, "en")
        else:
            # For Indian languages, prefer English clone with multilingual model
            # Better quality than Indian language clone recorded in noisy conditions
            voice_id = await get_elevenlabs_voice_id(user_id, "en")
            if not voice_id:
                # Fall back to language-specific clone only if no English clone
                lang_to_clone_key = {
                    "kn-IN": "kn", "hi-IN": "hi", "ta-IN": "ta", "te-IN": "te"
                }
                clone_key = lang_to_clone_key.get(language, "en")
                voice_id = await get_elevenlabs_voice_id(user_id, clone_key)

        if voice_id:
            elevenlabs_audio = await elevenlabs_tts(text, voice_id)
            if elevenlabs_audio:
                print(f"Using cloned voice for {language}")
                return elevenlabs_audio

    print(f"Using Sarvam TTS for language: {language}")
    return await sarvam_tts(text, language)

async def sarvam_tts(text: str, language_code: str) -> bytes:
    try:
        import time
        tts_start = time.time()
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
            tts_latency = round((time.time() - tts_start) * 1000)
            if response.status_code == 200:
                print(f"[LATENCY] TTS Sarvam: {tts_latency}ms | lang: {language_code}")
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

async def elevenlabs_tts(text: str, voice_id: str) -> bytes | None:
    """Generate speech using ElevenLabs with user's cloned voice."""
    try:
        import time
        tts_start = time.time()
        async with httpx.AsyncClient() as client:
            # Detect language to choose optimal model
            # eleven_multilingual_v2 handles Indian languages natively
            # eleven_flash_v2_5 is faster for English only
            text_language = detect_language(text)
            model_id = "eleven_flash_v2_5" if text_language == "en-IN" else "eleven_multilingual_v2"

            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": model_id,
                    "output_format": "mp3_44100_128",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                },
                timeout=30
            )
            tts_latency = round((time.time() - tts_start) * 1000)
            if response.status_code == 200:
                print(f"[LATENCY] TTS ElevenLabs: {tts_latency}ms | voice: {voice_id}")
                print(f"ElevenLabs first 4 bytes: {response.content[:4].hex()} len: {len(response.content)}")
                return response.content
            else:
                print(f"ElevenLabs TTS error: {response.status_code} {response.text}")
                return None
    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")
        return None

def pcm_to_mulaw(pcm_bytes: bytes) -> bytes:
    import audioop
    return audioop.lin2ulaw(pcm_bytes, 2)

def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    import audioop
    pcm = audioop.ulaw2lin(mulaw_bytes, 2)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    return buf.getvalue()

def mulaw_chunk_to_pcm(mulaw_bytes: bytes) -> bytes:
    import audioop
    return audioop.ulaw2lin(mulaw_bytes, 2)

async def transcribe(audio_bytes: bytes, language_hint: str = "en-IN") -> str:
    try:
        lang_map = {
            "en-IN": "en",
            "kn-IN": "kan",
            "hi-IN": "hi",
            "ta-IN": "ta",
            "te-IN": "te"
        }
        elevenlabs_lang = lang_map.get(language_hint, "en")

        import time
        stt_start = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={
                    "model_id": "scribe_v1",
                    "language_code": elevenlabs_lang,
                },
                timeout=30
            )
            stt_latency = round((time.time() - stt_start) * 1000)
            if response.status_code == 200:
                result = response.json()
                transcript = result.get("text", "")
                detected = result.get("language_code", elevenlabs_lang)
                print(f"[LATENCY] STT: {stt_latency}ms | lang: {detected} | text: '{transcript[:40]}'")
                return transcript
            else:
                print(f"ElevenLabs STT error: {response.status_code} {response.text}")
                return ""
    except Exception as e:
        print(f"ElevenLabs STT error: {e}")
        return ""

async def get_ai_response(conversation_history: list, business_context: str = "") -> str:
    try:
        system_content = business_context if business_context else (
            "You are UrVoice, an AI phone assistant for Indian users. "
            "Keep responses short, under 2 sentences. Be helpful and professional. "
            "Always respond in the same language the caller is using. "
            "Never ignore a language switch request."
        )

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Convert conversation history to Gemini format
        contents = []
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        import time
        llm_start = time.time()
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_content,
                    max_output_tokens=300,
                )
            )
        )
        llm_latency = round((time.time() - llm_start) * 1000)
        print(f"[LATENCY] LLM: {llm_latency}ms | response: '{response.text[:40]}'")
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""
