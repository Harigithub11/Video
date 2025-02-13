from TTS.api import TTS

MODEL_PATH = r"C:\Users\enguv\AppData\Local\tts\tts_models--en--ljspeech--tacotron2-DDC\tacotron2_DDC.pth"
CONFIG_PATH = r"C:\Users\enguv\AppData\Local\tts\tts_models--en--ljspeech--tacotron2-DDC\config.json"

try:
    tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH)
    
    # Generate speech and save to file
    tts.tts_to_file(text="Hello, this is a test.", file_path="test_output.wav")
    
    print("✅ Model loaded and speech generated: test_output.wav")
except Exception as e:
    print(f"❌ Error loading model: {e}")
