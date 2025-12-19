# app/tts_engine.py
import torch
from TTS.api import TTS

class CoquiTTSEngine:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ).to(device)

    def synthesize(self, text, speaker_wav, language, output_path):
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path
        )

    def save_voice_profile(self, speaker_wav, save_path):
        # Extract latents
        # Access model via synthesizer
        gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(audio_path=speaker_wav)
        
        # Save to file
        torch.save({
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }, save_path)
        
    def synthesize_from_profile(self, text, profile_path, language, output_path):
        # Load latents
        latents = torch.load(profile_path)
        
        # Inference with pre-computed latents
        out = self.tts.synthesizer.tts_model.inference(
            text,
            language,
            gpt_cond_latent=latents["gpt_cond_latent"],
            speaker_embedding=latents["speaker_embedding"]
        )
        
        # Save output audio
        # 'out' is usually a dictionary with 'wav' tensor for XTTS
        wav = out["wav"]
        
        # Convert to numpy for save_wav
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
            
        # Use synthesizer's save_wav to handle format/sample rate
        self.tts.synthesizer.save_wav(wav=wav, path=output_path)
