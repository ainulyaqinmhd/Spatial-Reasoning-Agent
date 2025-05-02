# Spatial Reasoning Imaging Agent for Education - HuggingFace Gradio version

import gradio as gr
from PIL import Image
import os
import subprocess
import pyttsx3
import torch
import torchaudio
import soundfile as sf
from vosk import Model, KaldiRecognizer
import wave
import json
from glob import glob

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prefer MPS (Apple Silicon GPU) if supported, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Silero STT model
stt_model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language="en",
    device=device,
)
read_batch, split_into_batches, read_audio, prepare_model_input = utils


# Run prompt with Ollama (text-only, Gemma3)
def ask_ollama(prompt, model="gemma3:4b-it-qat", image_path=None):
    cmd = ["ollama", "run", model]
    print("Running command:", " ".join(cmd))
    print("Prompt:", prompt)
    result = subprocess.run(
        cmd, input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print("STDOUT:", result.stdout.decode())
    print("STDERR:", result.stderr.decode())
    return result.stdout.decode()


# Text-to-Speech using pyttsx3 threaded
def speak_text(text):
    import threading
    import queue

    # Shared TTS queue to avoid overlapping threads
    if not hasattr(speak_text, "tts_queue"):
        speak_text.tts_queue = queue.Queue()

    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            for voice in engine.getProperty("voices"):
                if "Alex" in voice.name or "Samantha" in voice.name:
                    engine.setProperty("voice", voice.id)
                    break
            while not speak_text.tts_queue.empty():
                engine.say(speak_text.tts_queue.get())
            engine.runAndWait()
        except Exception:
            pass

    speak_text.tts_queue.put(text)
    threading.Thread(target=run_tts).start()


# Generate prompt


def generate_spatial_prompt(image_description: str):
    return f"""
You are a Spatial Reasoning Tutor skilled in interpreting architectural sketches and guiding users in transforming 2D isometric images into 3D models using Gravity Sketch in VR.

When the first image is uploaded, you must closely observe the visual details first before making any interpretation. Use step-by-step visual reasoning, and do not make assumptions that are not supported by the image itself.

⚠️ Avoid speculative phrases like "It seems to be..." or "This might be..." unless followed by a justification based on visible elements.

❌ Do NOT infer missing details, such as materials, context, or purpose, unless clearly shown.
✅ DO focus only on what is visible — lines, shapes, proportions, connections.

Your Task
Step 1: Careful Observation

Describe only what is clearly visible in the sketch.

Note the basic geometric forms (e.g., cubes, cylinders, planes) and how they connect.

Step 2: Spatial Breakdown

Identify parts such as floors, roofs, supports, walls, stairs — if and only if they are clearly represented.

Highlight any symmetry, repetition, or perspective cues that help locate parts in space.

Step 3: Tentative Interpretation

Based on the above, suggest (not assume) what type of structure it could be.

Use cautious language:

"This sketch appears to depict a structure with [feature], possibly suggesting a [type of building], due to [visible reason]."

Step 4: Confirm Intent
Ask the user:

"Would you like a step-by-step guide to model this structure in Gravity Sketch VR?"

If user replies Yes:
Provide a step-by-step modeling guide, including:

Suggested tools and brushes (e.g., Surface Tool, Revolve)

How to build the base forms

Recommendations for proportional alignment and navigating VR space

If user replies No:
Ask an alternative:

"Understood. Would you like feedback on design layout, proportions, or maybe suggestions for materials or lighting effects?"
"""


# Describe uploaded image
def describe_image(img: Image.Image) -> str:
    width, height = img.size
    return f"The image is {width}x{height} pixels, showing geometric elements arranged in a spatial layout."


image_desc = None


# Analyze uploaded image
def analyze_image(image, history):
    global image_desc
    if image is not None:
        path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
        image.save(path)
        image_desc = describe_image(image)
        prompt = generate_spatial_prompt(image_desc)
        response = ask_ollama(prompt)
        history.append({"role": "user", "content": "Uploaded Image"})
        history.append({"role": "assistant", "content": response})
        import time

        time.sleep(0.2)
        speak_text(response)
        print("Image analyzed, response generated:", response)
        return gr.update(value=history), "", history
    return history, "", history


# Chat interface logic
def chatbot_interface(image, message, history):
    global image_desc
    if isinstance(image, dict):
        image = image.get("image")
    if image is not None and not history:
        image.save(path)
        image_desc = describe_image(image)
        prompt = generate_spatial_prompt(image_desc)
        response = ask_ollama(prompt)
        history = [
            {"role": "user", "content": "Uploaded Image"},
            {"role": "assistant", "content": response},
        ]
        speak_text(response)
        return history, "", history
    note = f" (Refer to image: {image_desc})" if image_desc else ""
    prompt = f"Student: {message}{note}\nTutor:"
    response = ask_ollama(prompt)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    speak_text(response)
    return history, "", history


# Voice input transcription using Silero+soundfile
def transcribe_audio(audio_path) -> str:
    if not audio_path:
        return ""

    # Convert to standard WAV format readable by Vosk
    waveform, sample_rate = sf.read(audio_path)
    temp_wav_path = os.path.join(UPLOAD_FOLDER, "temp_input.wav")
    torchaudio.save(
        temp_wav_path, torch.tensor(waveform).float().unsqueeze(0), sample_rate
    )

    # Initialize Vosk model if not already loaded
    if not hasattr(transcribe_audio, "vosk_model"):
        transcribe_audio.vosk_model = Model(
            "vosk-model-small-en-us-0.15"
        )  # assumes you downloaded a model in en-us

    wf = wave.open(temp_wav_path, "rb")
    rec = KaldiRecognizer(transcribe_audio.vosk_model, wf.getframerate())

    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result += res.get("text", "") + " "
    res = json.loads(rec.FinalResult())
    result += res.get("text", "")

    return result.strip()


# Save chat history
def download_chat(history) -> str:
    path = os.path.join(UPLOAD_FOLDER, "chat_log.txt")
    with open(path, "w") as f:
        for msg in history:
            f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
    return path


# Gradio UI setup
demo = gr.Blocks()
with demo:
    gr.Markdown("## 🧠 Spatial Reasoning Imaging Agent for Education")

    history_state = gr.State([])
    with gr.Row():
        image_input = gr.Image(
            type="pil", label="Upload Sketch (JPG/PNG)", height=550, width="auto"
        )
        with gr.Column():
            chatbot = gr.Chatbot(label="Tutor Chat", type="messages")
            text_input = gr.Textbox(
                placeholder="Enter your question...", show_label=False
            )
            audio_input = gr.Microphone(
                label="🎙️ Speak", type="filepath", interactive=True
            )
    with gr.Row():
        download_btn = gr.Button("💾 Download Log")
        clear_btn = gr.Button("🧼 Clear Chat")

    image_input.change(
        analyze_image,
        [image_input, history_state],
        [chatbot, text_input, history_state],
    )
    text_input.submit(
        chatbot_interface,
        [image_input, text_input, history_state],
        [chatbot, text_input, history_state],
    )
    audio_input.change(
        lambda audio: chatbot_interface(
            image_input.value, transcribe_audio(audio), history_state.value
        ),
        [audio_input],
        [chatbot, text_input, history_state],
    )
    clear_btn.click(
        lambda history: ([], "", []),
        [history_state],
        [chatbot, text_input, history_state],
    )
    download_btn.click(download_chat, [history_state], gr.File())

    demo.launch()
