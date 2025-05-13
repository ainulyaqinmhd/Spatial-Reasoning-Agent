# Spatial Reasoning Conversation Assistant

A Gradio-based chatbot that integrates image analysis, speech recognition, and text-to-speech (TTS) to assist users in transforming 2D sketches into 3D models using Gravity Sketch VR.

## Features

- Image upload and analysis with enhanced preprocessing and multimodal prompts
- Speech recognition via Google API (online), Vosk (offline), Whisper (offline), or automatic selection
- Multiple TTS engines: gTTS, pyttsx3 (offline), and ElevenLabs (premium online)
- Voice assistant with wake word detection, continuous listening mode, and voice command processing
- IterDRAG iterative research mode for deep, iterative web search and summarization
- Interactive Gradio UI with voice assistant controls, iterative research toggle, and chat history saving
- Microphone testing and speech recognition engine selection
- Save conversation history to a text file for later review

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ainulyaqinmhd/Spatial-Reasoning-Agent.git
   cd Spatial-Reasoning-Agent
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. (Optional) Set environment variables for ElevenLabs API key and speech recognition engine:

   ```bash
   export ELEVENLABS_API_KEY="your_key_here"
   export RECOGNITION_ENGINE="auto"  # Options: "auto", "whisper", "vosk", "google"
   ```

## Usage

Run the application:

```bash
python main.py
```

Open your browser to the Gradio URL (usually `http://localhost:7860/`), or use the provided `share=True` link to access remotely.

## User Interface Overview

- Upload sketches (JPG/PNG) for spatial analysis
- Type messages or questions in the chatbox
- Use the microphone button or activate continuous voice assistant mode with a customizable wake word (default: "Hey Assistant")
- Toggle IterDRAG research mode for iterative deep research on queries
- Select speech recognition engine: Auto, Google, Vosk (offline), or Whisper (offline)
- Choose TTS voice: Default Female, Default Male, or Premium (ElevenLabs)
- Save conversation history to a text file
- Test microphone functionality with the provided button
- Clear conversation to start fresh

## Development Notes

- To test different speech recognition engines, edit the `RECOGNITION_ENGINE` variable in `main.py`
- To add or modify TTS engines, update the `setup_enhanced_tts()` function in `main.py`
- For extending image analysis capabilities, refer to the `analyze_image()` function in `main.py`
- Voice assistant functionality is implemented in the `VoiceListener` class
- IterDRAG iterative research mode is implemented in the `iterative_research_loop()` function
