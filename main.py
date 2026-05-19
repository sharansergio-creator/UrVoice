from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
def root():
    return {"status": "UrVoice backend running"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    response = VoiceResponse()
    response.say(
        "Hello, you have reached UrVoice. Our AI assistant will help you shortly.",
        voice="alice"
    )
    response.pause(length=1)
    response.say("Please leave your message after the tone.", voice="alice")
    return PlainTextResponse(str(response), media_type="application/xml")