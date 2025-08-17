import os
import time
import sounddevice as sd
import numpy as np
import requests
import google.generativeai as genai
import tempfile
import wave
from dotenv import load_dotenv


load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VOICE_ID_ENGLISH = os.getenv("VOICE_ID_ENGLISH")
VOICE_ID_ARABIC = os.getenv("VOICE_ID_ARABIC")
STT_MODEL = os.getenv("STT_MODEL", "scribe_v1")

SAMPLE_RATE = 16000
CHUNK_SEC = 2  


genai.configure(api_key=GEMINI_API_KEY)


def record_and_stream(duration=8):
    """Record microphone input in chunks and stream to STT."""
    print("Speak now... streaming transcription")
    all_text = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    stream.start()

    total_chunks = int(duration / CHUNK_SEC)
    for i in range(total_chunks):
        audio, _ = stream.read(int(SAMPLE_RATE * CHUNK_SEC))
        audio = audio.flatten()

        # save chunk as wav temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            with wave.open(tmpf.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            chunk_text = transcribe_with_elevenlabs(tmpf.name)
            if chunk_text:
                print(f"Partial ({i+1}): {chunk_text}")
                all_text.append(chunk_text)

    stream.stop()
    return " ".join(all_text)


def transcribe_with_elevenlabs(filename):
    """Send audio file chunk to ElevenLabs STT."""
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    with open(filename, "rb") as f:
        files = {"file": (filename, f, "audio/wav"), "model_id": (None, STT_MODEL)}
        r = requests.post(url, headers=headers, files=files)
    if r.status_code == 200:
        return r.json().get("text", "")
    else:
        print("STT error:", r.text)
        return ""


def detect_language(text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Detect if this text is English or Egyptian Arabic: '{text}'. Reply only with 'english' or 'arabic'."
    response = model.generate_content(prompt)
    lang = response.text.strip().lower()
    return "arabic" if "arabic" in lang else "english"


def generate_reply_with_gemini(prompt, language):
    model = genai.GenerativeModel("gemini-2.5-flash")
    lang_instruction = "Reply in Egyptian Arabic." if language == "arabic" else "Reply in English."
    response = model.generate_content(f"{lang_instruction} The user said: {prompt}")
    return response.text


def speak(text, language):
    voice_id = VOICE_ID_ARABIC if language == "arabic" else VOICE_ID_ENGLISH
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    r = requests.post(url, json=data, headers=headers)
    if r.status_code == 200:
        with open("response.mp3", "wb") as f:
            f.write(r.content)
        os.system("start response.mp3" if os.name == "nt" else "afplay response.mp3")
    else:
        print("TTS error:", r.text)


if __name__ == "__main__":
    text = record_and_stream(duration=8)
    print("Final transcription:", text)

    if text:
        lang = detect_language(text)
        reply = generate_reply_with_gemini(text, lang)
        print("AI:", reply)
        speak(reply, lang)
