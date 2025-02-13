import os
import re
import nltk
import aiohttp
import asyncio
import subprocess
from collections import Counter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
from nltk.corpus import stopwords
from TTS.api import TTS  # Coqui TTS
from io import BytesIO

# Download stopwords
nltk.download("stopwords")

# FastAPI app initialization
app = FastAPI()

# Enable CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "data"
AUDIO_DIR = "audio"
VIDEO_DIR = "public/videos"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# API keys
PIXABAY_API_KEY = "48738698-6ae327a6f8a04d813fa6c6101"
FREESOUND_API_KEY = "V3fdjHuyYMmUc1qmUFnZ0FOW6SBebGP980uryz4Y"

# Load Coqui TTS Model (Optimized)
MODEL_NAME = "tts_models/en/ljspeech/speedy-speech"

# Use a persistent model directory to avoid redownloading
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME.replace("/", "--"))

print("🔍 Checking for TTS model...")
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading TTS model...")
    coqui_tts = TTS(MODEL_NAME)
    coqui_tts.download()
    print("✅ TTS model downloaded.")

# Load the model
coqui_tts = TTS(MODEL_NAME)

# Text input request model
class TextRequest(BaseModel):
    text: str

# Extract text from an uploaded file (PDF/DOCX)
async def extract_text_from_file(file: UploadFile):
    """Extract text from an uploaded file (PDF or DOCX)."""
    content = await file.read()  # Read the file contents into memory
    file_stream = BytesIO(content)  # Convert to a seekable object

    if file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(file_stream)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.filename.endswith(".docx"):
        doc = Document(file_stream)  # Now, it's seekable
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return None  # Unsupported format
    return text

# Extract keywords from text
def extract_keywords(text, num_keywords=3):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    return [word for word, _ in Counter(filtered_words).most_common(num_keywords)]

# Fetch a video from Pixabay
async def fetch_video(keyword):
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&per_page=3"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    videos = data.get("hits", [])
                    if videos:
                        return videos[0]["videos"]["medium"]["url"]
    except Exception as e:
        print(f"❌ Error fetching video for {keyword}: {e}")
    return None

# Fetch background audio from Freesound
async def fetch_background_audio(keyword):
    url = f"https://freesound.org/apiv2/search/text/?query={keyword}&token={FREESOUND_API_KEY}&fields=previews"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    sounds = data.get("results", [])
                    if sounds:
                        return sounds[0]["previews"]["preview-hq-mp3"]
    except Exception as e:
        print(f"❌ Error fetching background audio for {keyword}: {e}")
    return None

# Convert text to speech using Coqui TTS
def text_to_speech(text, filename="output.wav"):
    audio_path = os.path.join(AUDIO_DIR, filename)
    try:
        print(f"🎤 Generating speech for text: {text[:100]}...")
        coqui_tts.tts_to_file(text=text, file_path=audio_path)
        print("✅ Speech generated successfully!")
        return audio_path
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        return None

# Generate video from text
@app.post("/generate_video/")
async def generate_video_endpoint(request: TextRequest):
    text = request.text
    return await generate_video(text)

# Generate video from uploaded file
@app.post("/generate_video_from_file/")
async def generate_video_from_file(file: UploadFile = File(...)):
    text = await extract_text_from_file(file)
    
    if not text:
        return JSONResponse(content={"error": "Unsupported file format or empty file"}, status_code=400)

    return await generate_video(text)

# Common function to generate video
async def generate_video(text):
    keywords = extract_keywords(text)

    # Fetch video and audio
    videos = await asyncio.gather(*(fetch_video(kw) for kw in keywords))
    background_audios = await asyncio.gather(*(fetch_background_audio(kw) for kw in keywords))
    speech_audio_path = text_to_speech(text)

    if not speech_audio_path:
        return JSONResponse(content={"error": "Failed to generate speech"}, status_code=500)
    if not videos[0]:
        return JSONResponse(content={"error": "No video found"}, status_code=500)
    if not background_audios[0]:
        return JSONResponse(content={"error": "No background audio found"}, status_code=500)

    # Generate final video
    output_video = os.path.join(VIDEO_DIR, "output.mp4")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", videos[0],
            "-i", speech_audio_path,
            "-i", background_audios[0],
            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_video
        ]
        subprocess.run(cmd, check=True)
        return FileResponse(output_video, media_type="video/mp4", filename="output.mp4")
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        return JSONResponse(content={"error": "Video creation failed"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "🚀 API is running successfully!"}
