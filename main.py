import os
import io
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai  # Modern Google GenAI SDK
from google.genai import types
from PIL import Image

app = FastAPI(title="AI Companion Backend")

# Initialize the client (automatically picks up GEMINI_API_KEY from environment)
client = genai.Client()

CRISIS_KEYWORDS = [
    "end my life", "suicide", "want to die", "hurt myself", 
    "ending it all", "don't want to live", "better off dead"
]

class ChatMessage(BaseModel):
    user_message: str

NORMAL_PROMPT = (
    "You are a handsome, warm, and highly supportive cartoon AI companion. "
    "You speak directly to a close friend. Be incredibly encouraging, cheerful, "
    "and lighthearted. Keep your responses brief (1-3 sentences) so they feel conversational."
)

CRISIS_PROMPT = (
    "CRISIS MODE ACTIVATED. The user is expressing thoughts of self-harm. "
    "Drop the casual cartoon character persona completely. Speak in an incredibly calm, "
    "grounded, and gentle voice. Do not encourage their despair. Remind them that they "
    "are safe, and instruct them clearly to tap the 'Get Help Now' button on their screen."
)

@app.post("/chat")
async def chat_with_companion(message: ChatMessage):
    user_text = message.user_message.lower()
    is_crisis = any(keyword in user_text for keyword in CRISIS_KEYWORDS)
    system_instructions = CRISIS_PROMPT if is_crisis else NORMAL_PROMPT

    try:
        # Using the standard modern endpoint format 'gemini-2.5-flash'
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=message.user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
            ),
        )
        
        ai_reply = response.text
        
        return {
            "ai_response": ai_reply,
            "trigger_crisis_ui": is_crisis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-avatar")
async def generate_avatar(file: UploadFile = File(...)):
    try:
        input_bytes = await file.read()
        img = Image.open(io.BytesIO(input_bytes))
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
