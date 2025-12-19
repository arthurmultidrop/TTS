# app/main.py
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import uuid
import os
import tempfile
from app.tts_engine import CoquiTTSEngine

app = FastAPI()
tts_engine = CoquiTTSEngine()

@app.post("/tts")
def generate_tts(
    text: str = Form(...),
    language: str = Form("pt"),
    speaker_wav: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    speaker_path = os.path.join(temp_dir, f"{job_id}_speaker.wav")
    
    # Save output to project folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{job_id}_output.wav")

    with open(speaker_path, "wb") as f:
        shutil.copyfileobj(speaker_wav.file, f)

    tts_engine.synthesize(
        text=text,
        speaker_wav=speaker_path,
        language=language,
        output_path=output_path
    )

    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

class TTSRequest(BaseModel):
    text: str
    voice_id: str
    language: str = "pt"

@app.post("/voices")
def create_voice(speaker_wav: UploadFile = File(...)):
    voice_id = str(uuid.uuid4())
    voices_dir = "voices"
    os.makedirs(voices_dir, exist_ok=True)
    
    # Save temp wav for processing
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{voice_id}.wav")
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(speaker_wav.file, f)
        
    profile_path = os.path.join(voices_dir, f"{voice_id}.pth")
    tts_engine.save_voice_profile(temp_path, profile_path)
    
    # Compute stats or checking if created?
    # os.remove(temp_path) # Clean up temp
    
    return {"voice_id": voice_id}

@app.post("/tts_with_profile")
def generate_tts_cached(request: TTSRequest):
    job_id = str(uuid.uuid4())
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{job_id}.wav")
    
    profile_path = os.path.join("voices", f"{request.voice_id}.pth")
    
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="Voice profile not found")
        
    tts_engine.synthesize_from_profile(
        text=request.text,
        profile_path=profile_path,
        language=request.language,
        output_path=output_path
    )
    
    return {
        "jobId": job_id,
        "audioPath": output_path
    }
