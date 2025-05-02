# Spatial Reasoning Conversation Assistant

A Gradio‐based chatbot that combines image analysis, speech recognition, and TTS to guide users in converting 2D sketches into 3D models in Gravity Sketch VR.

## Features

- **Image upload & analysis** using OpenAI Whisper / Vosk
- **Speech recognition** via Google API, Vosk (offline), or Whisper (offline)
- **Text‐to‐Speech** engines: gTTS, pyttsx3, or ElevenLabs
- **Interactive UI** with Gradio for cross‐platform demo

## Installation

1. Clone this repo:
   ```
   git clone https://github.com/ainulyaqinmhd/Spatial-Reasoning-Agent.git
   cd Spatial-Reasoning-Agent
   ````

2. Create a virtual environment and install dependencies:

   ```
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. (Optional) Set environment variables for ElevenLabs:

   ```
   export ELEVENLABS_API_KEY="your_key_here"
   export RECOGNITION_ENGINE="auto"  # or "whisper", "vosk", "google"
   ```

## Usage

```
python app.py
```

Then point your browser to the Gradio URL (usually `http://localhost:7860/`), or share it via the `share=True` link.

## Development

* **Testing different engines:** edit `RECOGNITION_ENGINE` in `app.py`
* **Adding new TTS:** modify `setup_enhanced_tts()`
* **Extending image analysis:** see the `analyze_image()` function
