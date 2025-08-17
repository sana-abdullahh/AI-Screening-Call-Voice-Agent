import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import traceback
import google.generativeai as genai

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VOICE_ID_ENGLISH = os.getenv("VOICE_ID_ENGLISH")
VOICE_ID_ARABIC = os.getenv("VOICE_ID_ARABIC")
STT_MODEL = os.getenv("STT_MODEL")

# initialize gemini
genai.configure(api_key=GEMINI_API_KEY)


def record_audio(filename="input.wav", duration=5, fs=16000):
    start = time.time()
    print("Recording... please speak")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, fs, audio)
    elapsed = time.time() - start
    print(f"Recording saved as {filename} ({elapsed:.2f}s)")
    return elapsed

def transcribe_with_elevenlabs(filename="input.wav"):
    start = time.time()
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    with open(filename, "rb") as f:
        files = {
            "file": (filename, f, "audio/wav"),
            "model_id": (None, STT_MODEL)
        }
        response = requests.post(url, headers=headers, files=files)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e} - {response.text}")
        return "", time.time() - start
    result = response.json()
    elapsed = time.time() - start
    print(f"STT took {elapsed:.2f}s")
    return result.get("text", ""), elapsed

def detect_language(text):
    start = time.time()
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Detect if this text is English or Egyptian Arabic: '{text}'. Reply only with 'english' or 'arabic'."
    response = model.generate_content(prompt)
    elapsed = time.time() - start
    print(f"Language detection took {elapsed:.2f}s")
    lang = response.text.strip().lower()
    if "arabic" in lang:
        return "arabic", elapsed
    return "english", elapsed

def generate_reply_with_gemini(prompt, language):
    start = time.time()
    model = genai.GenerativeModel("gemini-1.5-flash")
    lang_instruction = "Reply in Egyptian Arabic." if language == "arabic" else "Reply in English."
    response = model.generate_content(f"{lang_instruction} The user said: {prompt}")
    elapsed = time.time() - start
    print(f"LLM generation took {elapsed:.2f}s")
    return response.text, elapsed

def speak(text, language):
    start = time.time()
    voice_id = VOICE_ID_ARABIC if language == "arabic" else VOICE_ID_ENGLISH
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    r = requests.post(url, json=data, headers=headers)
    r.raise_for_status()
    with open("response.mp3", "wb") as f:
        f.write(r.content)
    os.system("start response.mp3" if os.name == "nt" else "afplay response.mp3")
    elapsed = time.time() - start
    print(f"TTS took {elapsed:.2f}s")
    return elapsed

if __name__ == "__main__":
    try:
        total_start = time.time()

        t_record = record_audio()
        text, t_stt = transcribe_with_elevenlabs()

        if text:
            print("🗣 You said:", text)
            lang, t_lang_detect = detect_language(text)
            print(f"Detected language: {lang}")
            ai_reply, t_llm = generate_reply_with_gemini(text, lang)
            print("AI:", ai_reply)
            t_tts = speak(ai_reply, lang)

            total_elapsed = time.time() - total_start
            print("\nLatency breakdown:")
            print(f"  Recording: {t_record:.2f}s")
            print(f"  STT: {t_stt:.2f}s")
            print(f"  Language detection: {t_lang_detect:.2f}s")
            print(f"  LLM generation: {t_llm:.2f}s")
            print(f"  TTS: {t_tts:.2f}s")
            print(f"  TOTAL: {total_elapsed:.2f}s")
        else:
            print(" No transcription result.")
    except Exception as e:
        traceback.print_exc()