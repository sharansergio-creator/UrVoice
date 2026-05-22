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


def _text(soup: BeautifulSoup, *selectors) -> str:
    """Return stripped text of the first matching selector."""
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            return tag.get_text(" ", strip=True)
    return ""


def _all_text(soup: BeautifulSoup, *selectors) -> list[str]:
    """Return a list of stripped text for every matching element."""
    results = []
    for sel in selectors:
        for tag in soup.select(sel):
            t = tag.get_text(" ", strip=True)
            if t:
                results.append(t)
    return results


async def _scrape_gbp(gbp_url: str) -> dict:
    """Extract structured data from a Google Business Profile share URL."""
    result = {"businessName": "", "address": "", "phone": "", "hours": "",
               "rating": "", "category": ""}
    try:
        html = await _fetch_html(gbp_url)
        soup = BeautifulSoup(html, "lxml")

        # Business name — og:title or <title>
        og_title = soup.find("meta", property="og:title")
        result["businessName"] = (
            og_title["content"].strip()
            if og_title and og_title.get("content")
            else soup.title.string.strip() if soup.title else ""
        )

        full_text = soup.get_text(" ", strip=True)

        # Address — look for structured microdata or heuristic
        addr_tag = soup.find(attrs={"itemprop": "address"})
        if addr_tag:
            result["address"] = addr_tag.get_text(" ", strip=True)
        else:
            # Heuristic: first occurrence of a postcode-like pattern
            m = re.search(r"[\w\s,]+(\d{6}|\d{5}(-\d{4})?)[\w\s,]*", full_text)
            if m:
                result["address"] = m.group(0).strip()[:120]

        # Phone
        phone_tag = soup.find(attrs={"itemprop": "telephone"})
        if phone_tag:
            result["phone"] = phone_tag.get_text(" ", strip=True)
        else:
            m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", full_text)
            if m:
                result["phone"] = m.group(1).strip()

        # Rating
        rating_tag = soup.find(attrs={"itemprop": "ratingValue"})
        review_tag = soup.find(attrs={"itemprop": "reviewCount"})
        if rating_tag:
            rating = rating_tag.get("content") or rating_tag.get_text(strip=True)
            reviews = ""
            if review_tag:
                reviews = review_tag.get("content") or review_tag.get_text(strip=True)
            result["rating"] = f"{rating} ({reviews} reviews)".strip() if reviews else rating
        else:
            m = re.search(r"(\d\.\d)\s*[\u2605★]?\s*[\(]?(\d[\d,]+)\s*reviews?", full_text, re.I)
            if m:
                result["rating"] = f"{m.group(1)} ({m.group(2)} reviews)"

        # Business hours — og:description often contains them
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            result["hours"] = og_desc["content"].strip()[:300]

        # Category
        cat_tag = soup.find(attrs={"itemprop": "servesCuisine"}) or \
                  soup.find(attrs={"itemprop": "category"})
        if cat_tag:
            result["category"] = cat_tag.get_text(" ", strip=True)

    except Exception as e:
        print(f"GBP scrape error: {e}")
    return result


async def _groq_web_search(business_name: str, website_url: str, gbp_url: str) -> dict:
    """Use Groq compound-beta (web search) to find missing business info online."""
    fields = {"address": "", "phone": "", "hours": "", "about": "", "services": "", "pricing": ""}
    if not GROQ_API_KEY:
        return fields
    query = business_name or website_url or gbp_url
    if not query:
        return fields
    ref = website_url or gbp_url
    try:
        async with httpx.AsyncClient(timeout=50) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "compound-beta",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f'Find details for the business "{query}"'
                                + (f" at {ref}" if ref else "") + ". "
                                "I need: full address, phone number, business hours, "
                                "about/description of what the business does, "
                                "services or products offered, and pricing details. "
                                "Return ONLY valid JSON with keys: "
                                "address, phone, hours, about, services, pricing. "
                                "Use empty string for fields not found. No markdown, just JSON."
                            ),
                        }
                    ],
                    "max_tokens": 700,
                    "temperature": 0.1,
                },
            )
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            for key in fields:
                val = parsed.get(key, "")
                if val:
                    fields[key] = str(val)[:600]
        print(f"Groq web search result keys: {[k for k, v in fields.items() if v]}")
    except Exception as e:
        print(f"Groq web search error: {e}")
    return fields


async def _extract_with_groq(full_text: str) -> dict:
    """Use Groq LLM to extract about/services/pricing from raw website text."""
    extracted = {"about": "", "services": "", "pricing": ""}
    if not GROQ_API_KEY or not full_text.strip():
        return extracted
    trimmed = full_text[:4000]
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract business info from website text. "
                                "Return ONLY a JSON object with exactly these keys: "
                                "\"about\" (2-3 sentences describing what the business does), "
                                "\"services\" (comma-separated list of services or products), "
                                "\"pricing\" (pricing details if present, else empty string). "
                                "No extra text, no markdown, just the JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Extract business info:\n\n{trimmed}",
                        },
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
            )
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            extracted["about"]    = str(parsed.get("about",    ""))[:600]
            extracted["services"] = str(parsed.get("services", ""))[:400]
            extracted["pricing"]  = str(parsed.get("pricing",  ""))[:200]
    except Exception as e:
        print(f"Groq extraction error: {e}")
    return extracted


async def _scrape_website(website_url: str) -> dict:
    """Extract about/services/pricing/contact/FAQ text from a website."""
    result = {"about": "", "services": "", "pricing": "", "contact": "", "faqs": ""}
    pages_html: list[str] = []

    # Fetch homepage
    try:
        homepage_html = await _fetch_html(website_url)
        pages_html.append(homepage_html)
    except Exception as e:
        print(f"Website homepage fetch error: {e}")
        return result

    # Try common sub-pages
    base = website_url.rstrip("/")
    for slug in ("/about", "/about-us", "/services", "/our-services", "/faq", "/faqs"):
        try:
            html = await _fetch_html(f"{base}{slug}")
            pages_html.append(html)
        except Exception:
            pass

    combined_soup = BeautifulSoup("".join(pages_html), "lxml")

    # Remove nav / header / footer / scripts / styles noise
    for tag in combined_soup.select("nav, header, footer, script, style, noscript"):
        tag.decompose()

    full_text = combined_soup.get_text(" ", strip=True)

    # About
    about_section = combined_soup.find(
        lambda t: t.name in ("section", "div", "article")
        and re.search(r"about", t.get("id", "") + " ".join(t.get("class", [])), re.I)
    )
    if about_section:
        result["about"] = about_section.get_text(" ", strip=True)[:600]
    else:
        m = re.search(r"(?i)about\s+us[:\-]?\s*(.{50,400})", full_text)
        if m:
            result["about"] = m.group(1).strip()[:400]

    # Services — look for lists inside a services section
    svc_items = _all_text(
        combined_soup,
        "[id*='service'] li, [class*='service'] li",
        "[id*='Service'] li, [class*='Service'] li",
    )
    if svc_items:
        result["services"] = "; ".join(svc_items[:15])
    else:
        m = re.search(r"(?i)services?[:\-]?\s*(.{30,400})", full_text)
        if m:
            result["services"] = m.group(1).strip()[:400]

    # Pricing
    prices = re.findall(r"(?:Rs\.?|INR|\$|₹)\s*[\d,]+(?:\.\d{1,2})?", full_text)
    if prices:
        result["pricing"] = ", ".join(dict.fromkeys(prices[:10]))
    else:
        m = re.search(r"(?i)pric(?:e|ing)[:\-]?\s*(.{20,200})", full_text)
        if m:
            result["pricing"] = m.group(1).strip()[:200]

    # Contact
    phones = re.findall(r"(\+?\d[\d\s\-().]{7,}\d)", full_text)
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", full_text)
    contact_parts = []
    if phones:
        contact_parts.append("Ph: " + ", ".join(dict.fromkeys(phones[:3])))
    if emails:
        contact_parts.append("Email: " + ", ".join(dict.fromkeys(emails[:3])))
    result["contact"] = "  ".join(contact_parts)

    # FAQs
    faq_items = _all_text(
        combined_soup,
        "[id*='faq'] h3, [id*='faq'] h4, [class*='faq'] h3, [class*='faq'] h4",
        "[id*='FAQ'] h3, [id*='FAQ'] h4, [class*='FAQ'] h3, [class*='FAQ'] h4",
    )
    if faq_items:
        result["faqs"] = "; ".join(faq_items[:10])

    # If heuristics couldn't extract about/services/pricing, use Groq LLM
    if not result["about"] or not result["services"]:
        groq_data = await _extract_with_groq(full_text)
        if not result["about"] and groq_data["about"]:
            result["about"] = groq_data["about"]
        if not result["services"] and groq_data["services"]:
            result["services"] = groq_data["services"]
        if not result["pricing"] and groq_data["pricing"]:
            result["pricing"] = groq_data["pricing"]

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
        "address":       gbp_data.get("address") or "",
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

    # If key fields are still missing, use Groq web search to find them online
    missing = not result["address"] or not result["about"] or not result["services"] or not result["pricing"]
    if missing:
        search_result = await _groq_web_search(
            result.get("businessName", ""),
            body.website_url,
            body.gbp_url,
        )
        for field in ("address", "phone", "hours", "about", "services", "pricing"):
            if not result.get(field) and search_result.get(field):
                result[field] = search_result[field]

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