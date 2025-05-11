import gradio as gr
from PIL import Image
import os
import subprocess
import threading
import queue
import time
import numpy as np
import atexit
import tempfile
from gtts import gTTS
import pygame  # for audio playback (cross-platform)
import speech_recognition as sr  # Google Speech Recognition
import traceback
import json
import soundfile as sf  # For audio file analysis
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("spatial_agent.log")],
)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Initialize speech recognizer with more robust settings and comments explaining each parameter
recognizer = sr.Recognizer()
recognizer.energy_threshold = 100  # Adjust based on your microphone/environment. Lower values may detect quieter sounds, while higher values may require louder input.
recognizer.dynamic_energy_threshold = True  # Enable dynamic energy threshold to adjust the recognizer's sensitivity automatically.
recognizer.pause_threshold = 1.0  # Increased from 0.8 to 1.0 seconds. This parameter determines how long of a pause is considered the end of a phrase.
recognizer.phrase_threshold = 0.3  # Adjusts how confident the recognizer must be before considering a partial match as a complete phrase.
recognizer.non_speaking_duration = 0.8  # Increased from 0.5 to 0.8. This parameter determines how long of a non-speech segment is considered part of the previous speech.
# Add adjustable language for international support
SPEECH_LANGUAGE = (
    "en-US"  # Default to English, but this can be changed to support other languages.
)

# Store conversation context
conversation_context = {
    "current_topic": None,
    "last_question": None,
    "greeting_done": False,
    "image_uploaded": False,
    "image_analyzed": False,  # Track if an image has been analyzed
}

# Create a single global TTS queue
tts_queue = queue.Queue()
tts_thread_active = False

# Define global variables
tts_type = "gtts"  # Default TTS engine
tts_engine = None
use_elevenlabs = False
elevenlabs_api_key = None  # Add your key if using ElevenLabs


# More natural TTS with pyttsx3 (offline) or ElevenLabs (online)
def setup_enhanced_tts():
    """Set up enhanced text-to-speech with fallback options"""
    global tts_engine, use_elevenlabs, tts_type

    # Configuration flags
    use_elevenlabs = False  # Set to True to use ElevenLabs premium voices
    elevenlabs_api_key = None  # Add your key if using ElevenLabs

    # Try to set up pyttsx3 (offline TTS)
    try:
        import pyttsx3

        tts_engine = pyttsx3.init()

        # Configure voice properties
        voices = tts_engine.getProperty("voices")
        # Select a voice - index 0 is usually male, 1 is female
        tts_engine.setProperty("voice", voices[1].id)  # Female voice
        # Set speaking rate (default is 200)
        tts_engine.setProperty("rate", 175)  # Slightly slower for clarity

        logger.info("pyttsx3 TTS engine initialized successfully")
        tts_type = "pyttsx3"
        return "pyttsx3"
    except Exception as e:
        logger.warning(f"Could not initialize pyttsx3: {e}")

    # Fall back to ElevenLabs if configured
    if use_elevenlabs and elevenlabs_api_key:
        try:
            import requests

            # Test the API connection
            headers = {
                "xi-api-key": elevenlabs_api_key,
                "Content-Type": "application/json",
            }
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices", headers=headers
            )
            if response.status_code == 200:
                logger.info("ElevenLabs TTS connected successfully")
                tts_type = "elevenlabs"
                return "elevenlabs"
            else:
                logger.warning(f"ElevenLabs API error: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not initialize ElevenLabs: {e}")

    # Final fallback to gTTS (already implemented)
    logger.info("Using gTTS as fallback TTS engine")
    tts_type = "gtts"
    return "gtts"


# Start the TTS worker thread
def start_tts_thread():
    global tts_thread_active
    if not tts_thread_active:
        tts_thread_active = True
        threading.Thread(target=tts_worker, daemon=True).start()
        logger.info("TTS thread started")


def speak_text(text, interrupt=True):
    """Speak text using the configured TTS engine"""
    global tts_engine, tts_thread_active, tts_type
    if not text or text.strip() == "":
        return
    # Option to interrupt current speech
    if interrupt and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    # Start TTS thread if not active
    if not tts_thread_active:
        start_tts_thread()

    # Split long texts into manageable chunks
    chunk_size = 500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        # Add to queue based on engine type
        if tts_type == "pyttsx3":
            tts_queue.put(("pyttsx3", chunk))
        elif tts_type == "elevenlabs":
            tts_queue.put(("elevenlabs", chunk))
        else:
            tts_queue.put(("gtts", chunk))

    logger.info(f"Added text to TTS queue: {text[:50]}..." if len(text) > 50 else text)


# Improved TTS worker that handles multiple engines
def tts_worker():
    """Worker thread that processes TTS requests from queue"""
    global tts_thread_active, tts_engine, use_elevenlabs, elevenlabs_api_key

    while tts_thread_active:
        try:
            # Wait for up to 0.5 seconds for a message
            item = tts_queue.get(timeout=0.5)
            if not item:
                tts_queue.task_done()
                continue

            engine_type, text = item

            # Handle pyttsx3 (offline TTS)
            if engine_type == "pyttsx3":
                try:
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                    tts_queue.task_done()
                    continue
                except Exception as e:
                    logger.error(f"pyttsx3 error: {e}")
                    # Fall through to next option

            # Handle ElevenLabs (premium online TTS)
            if engine_type == "elevenlabs" and use_elevenlabs and elevenlabs_api_key:
                try:
                    import requests

                    # ElevenLabs API call
                    headers = {
                        "xi-api-key": elevenlabs_api_key,
                        "Content-Type": "application/json",
                    }

                    data = {
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    }

                    # Using Rachel voice (change voice_id as needed)
                    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice

                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                        headers=headers,
                        json=data,
                    )

                    if response.status_code == 200:
                        # Save audio to temp file
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".mp3"
                        )
                        temp_filename = temp_file.name
                        temp_file.write(response.content)
                        temp_file.close()

                        # Play with pygame
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                        # Clean up
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass

                        tts_queue.task_done()
                        continue
                    else:
                        logger.error(f"ElevenLabs API error: {response.status_code}")
                        # Fall through to gTTS
                except Exception as e:
                    logger.error(f"ElevenLabs error: {e}")
                    # Fall through to gTTS

            # Fallback to gTTS (already implemented)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_filename = temp_file.name
            temp_file.close()

            # Generate speech with gTTS
            tts = gTTS(text=text, lang="en")
            tts.save(temp_filename)

            # Play the audio with pygame
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Clean up the temporary file after playing
            try:
                os.unlink(temp_filename)
            except:
                pass

            tts_queue.task_done()

        except queue.Empty:
            continue  # No message, continue waiting
        except Exception as e:
            logger.error(f"TTS error: {e}")
            continue


# Run prompt with Ollama (text-only, gemma3:4b-it-qat)
def ask_ollama(prompt, model="gemma3", system_prompt=None):
    cmd = ["ollama", "run", model]
    logger.info("Running command: %s", " ".join(cmd))

    # Prepare the input
    if system_prompt:
        input_data = {"system": system_prompt, "prompt": prompt}
        input_str = json.dumps(input_data)
    else:
        input_str = prompt

    logger.info("Prompt: %s", input_str)

    try:
        result = subprocess.run(
            cmd,
            input=input_str.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        response = result.stdout.decode()
        logger.info("STDOUT: %s", response)
        logger.info("STDERR: %s", result.stderr.decode())
        return response
    except Exception as e:
        logger.error(f"Error running Ollama: {e}")
        return f"I encountered an error while processing your request: {str(e)}"


# Generate better spatial reasoning prompts
def generate_spatial_prompt(image_description: str):
    return f"""
You are analyzing an architectural or spatial design sketch. The image shows a spatial structure or design, and I need you to analyze it carefully. Be precise about what you can actually see, not what you imagine could be there.

Focus on (Keep the answer no more than 150 words to maintain the flow of conversation):
1. Basic geometric elements visible in the image (lines, planes, volumes)
2. Spatial relationships between elements
3. Any visible textures, materials, or surface treatments
4. Perspective and depth cues
5. Proportions and scale indicators

If you can identify the likely type of structure or design (e.g., building, interior space, furniture), explain what visual elements support that identification.

After your analysis, ask whether it is necessary to provide suggestions for how this design could be modeled in 3D using Gravity Sketch VR, focusing on the approach to constructing the main elements.

Keep your response conversational, engaging, and educational - as if you're a helpful spatial reasoning tutor guiding a student.

"""


# Enhanced system prompt for more spatial awareness
SPATIAL_SYSTEM_PROMPT = """
You are a specialized spatial reasoning tutor with expertise in architectural visualization, 3D modeling, and visual-spatial analysis. You have the following capabilities and characteristics:

1. VISUAL ANALYSIS: You can carefully analyze images of architectural drawings, sketches, and spatial designs to identify elements, relationships, and design principles.

2. SPATIAL EDUCATION: You explain complex spatial concepts in accessible language with helpful analogies.

3. MODELING GUIDANCE: You provide step-by-step guidance for translating 2D representations into 3D models using tools like Gravity Sketch VR.

4. CONTEXTUAL AWARENESS: Your responses acknowledge the visual information present in uploaded images, and you refer to specific visible elements when discussing them.

5. COMMUNICATION STYLE: Your tone is:
   - Warm and encouraging
   - Conversational but knowledgeable
   - Focused on spatial thinking development
   - Precise about visual observations (avoiding speculation)
   - Patient with spatial learning challenges

Maintain a balance between technical precision and conversational engagement in your responses.

Keep the answer no more than 150 words to maintain the flow of conversation.
"""

# Define conversation system prompt
CONVERSATION_SYSTEM_PROMPT = """
You are an intelligent and helpful spatial reasoning tutor. Your goal is to assist users in understanding 3D spatial concepts, architectural principles, and help them translate 2D sketches into 3D thinking.

When responding to queries:
1. Be precise but conversational in your explanations
2. Reference specific visual elements if an image has been shared
3. Provide educational insights that help develop spatial thinking
4. Always maintain a supportive, encouraging tone

If a question is unclear or lacks context, politely ask for clarification rather than making assumptions.
"""


# Updated conversation handling for multimodal context
def maintain_conversation_context(message, image_path, conversation_context):
    """Enhance the conversation context with spatial references and visual continuity"""

    # Set default values if not present
    if "visual_references" not in conversation_context:
        conversation_context["visual_references"] = []

    # Add spatial awareness triggers
    spatial_keywords = [
        "left",
        "right",
        "above",
        "below",
        "front",
        "back",
        "perspective",
        "angle",
        "view",
        "scale",
        "proportion",
        "distance",
        "space",
        "volume",
        "shape",
        "form",
    ]

    # Check if message contains spatial references
    contains_spatial_reference = any(
        keyword in message.lower() for keyword in spatial_keywords
    )

    if contains_spatial_reference:
        conversation_context["spatial_discussion"] = True

    # Record if user is asking about specific visual elements
    if image_path and any(
        x in message.lower()
        for x in ["what's this", "what is this", "this part", "that element"]
    ):
        # We should handle deictic references (this/that/there) with more sophistication
        conversation_context["deictic_reference"] = True

    return conversation_context


# Describe uploaded image with more detail
def describe_image(img: Image.Image) -> str:
    width, height = img.size
    # Get more information about the image to help the model understand it better
    mode = img.mode
    format_type = img.format if hasattr(img, "format") else "Unknown"

    # Additional analysis could be added here based on your needs
    return f"The image is {width}x{height} pixels, {mode} mode, {format_type} format, showing geometric elements arranged in a spatial layout."


# Global variable to store image description
image_desc = None


# Generate a welcome message
def get_welcome_message():
    welcome_messages = [
        "Hi there! I'm your spatial reasoning assistant. Upload a sketch or ask me a question to get started!",
        "Hello! I'm ready to help you transform 2D sketches into 3D models. What would you like to work on today?",
        "Welcome! I can help analyze your architectural sketches and guide you in creating 3D models with Gravity Sketch. How can I assist you?",
    ]
    return np.random.choice(welcome_messages)


def analyze_image(image, history):
    global image_desc
    global conversation_context

    if image is not None:
        try:
            path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
            image.save(path)
            image_desc = describe_image(image)

            # Create multimodal prompt with Ollama/Gemma3
            prompt = generate_spatial_prompt(image_desc)

            # Send the image with the text prompt using Ollama's multimodal capabilities
            response = ask_ollama_with_image(prompt, path)

            # Update conversation context
            conversation_context["image_uploaded"] = True
            conversation_context["current_topic"] = "spatial_analysis"
            conversation_context["image_path"] = (
                path  # Store image path for future reference
            )

            history.append({"role": "user", "content": "Uploaded Image"})
            history.append({"role": "assistant", "content": response})

            # Speak the response
            speak_text(response)

            logger.info("Image analyzed, response generated: %s", response)
            return gr.update(value=history), "", history
        except Exception as e:
            error_msg = f"I ran into a problem analyzing that image. Could you try uploading it again or maybe a different one?"
            logger.error(f"Error analyzing image: {str(e)}")
            traceback.print_exc()
            history.append({"role": "assistant", "content": error_msg})
            return gr.update(value=history), "", history
    return history, "", history


# New function to send both text and image to Ollama with Gemma3
def ask_ollama_with_image(
    prompt, image_path, model="llama3.2-vision", system_prompt=None
):
    """
    Send multimodal prompt (text + image) to Ollama using Gemma3 vision model
    """
    logger.info(f"Running multimodal request with image: {image_path}")

    try:
        # Prepare the base64 encoded image
        with open(image_path, "rb") as img_file:
            import base64

            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Format the multimodal request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": [img_base64],
        }

        if system_prompt:
            request_data["system"] = system_prompt

        # Use requests to post to Ollama API directly
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate", json=request_data
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received from model")
        else:
            logger.error(
                f"Error from Ollama API: {response.status_code}, {response.text}"
            )
            return f"Error communicating with the model: {response.status_code}"

    except Exception as e:
        logger.error(f"Error in multimodal request: {e}")
        return f"I encountered an error while processing the image: {str(e)}"


# Update chatbot_interface to handle multimodal conversations
def chatbot_interface(image, message, history):
    global image_desc
    global conversation_context

    # Skip empty messages
    if not message or message.strip() == "":
        return history, "", history

    try:
        if isinstance(image, dict):
            image = image.get("image")

        # First-time greeting if no history and no message
        if not history and not conversation_context["greeting_done"] and not image:
            welcome = get_welcome_message()
            history = [{"role": "assistant", "content": welcome}]
            conversation_context["greeting_done"] = True
            speak_text(welcome)
            return history, "", history

        # Process image if available and no history
        if image is not None and not history:
            return analyze_image(image, history)

        # Build conversation history for context
        context = ""
        if len(history) > 0:
            for msg in history[-6:]:  # Last 6 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"

        # Determine if we should include the image in this response
        should_include_image = (
            image is not None or conversation_context.get("image_path") is not None
        )

        image_path = None
        if image is not None:
            # Save new image
            path = os.path.join(UPLOAD_FOLDER, "current.png")
            image.save(path)
            image_path = path
            image_desc = describe_image(image)
        elif conversation_context.get("image_path"):
            # Use previously uploaded image
            image_path = conversation_context.get("image_path")

        conversation_context["last_question"] = message

        prompt = f"""
{context}

User: {message}
Assistant:"""

        # Choose appropriate function based on whether image should be included
        if should_include_image and image_path:
            response = ask_ollama_with_image(
                prompt, image_path, system_prompt=CONVERSATION_SYSTEM_PROMPT
            )
        else:
            response = ask_ollama(prompt, system_prompt=CONVERSATION_SYSTEM_PROMPT)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        # Always speak the response
        speak_text(response)

        return history, "", history
    except Exception as e:
        error_msg = f"Sorry, I had trouble processing that. Could you try rephrasing your question?"
        logger.error(f"Error processing message: {str(e)}")
        traceback.print_exc()
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "", history


# Import for offline voice recognition
try:
    import vosk

    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning(
        "Vosk not installed. Falling back to Google Speech Recognition only."
    )

# Add Whisper if installed
try:
    import whisper

    WHISPER_AVAILABLE = True
    try:
        # Load a small model by default
        WHISPER_MODEL = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully")
    except AttributeError:
        WHISPER_AVAILABLE = False
        logger.error(
            "Whisper module does not have 'load_model'. Please ensure you have installed the correct OpenAI whisper package: 'pip install -U openai-whisper'"
        )
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning(
        "Whisper not installed. It won't be available as a recognition option."
    )

# Voice recognition configuration
RECOGNITION_ENGINE = "auto"  # Options: "google", "vosk", "whisper", "auto"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # Path to Vosk model

# Initialize Vosk if available
if VOSK_AVAILABLE:
    try:
        if os.path.exists(VOSK_MODEL_PATH):
            vosk_model = vosk.Model(VOSK_MODEL_PATH)
            logger.info(f"Vosk model loaded from {VOSK_MODEL_PATH}")
        else:
            logger.warning(
                f"Vosk model not found at {VOSK_MODEL_PATH}. Vosk won't be available."
            )
            VOSK_AVAILABLE = False
    except Exception as e:
        logger.error(f"Error loading Vosk model: {e}")
        VOSK_AVAILABLE = False


# Enhanced image preprocessing for better model interpretation
def preprocess_image_for_analysis(image):
    """Enhance sketches and drawings for better multimodal model understanding"""
    try:
        import cv2
        import numpy as np
        from PIL import ImageEnhance, ImageOps

        # Convert PIL image to CV2 format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Create enhanced versions optimized for line detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Basic adaptive thresholding for line detection
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Edge detection to highlight structural elements
        edges = cv2.Canny(gray, 50, 150)

        # Save processed versions
        processed_path = os.path.join(UPLOAD_FOLDER, "processed")
        os.makedirs(processed_path, exist_ok=True)

        cv2.imwrite(os.path.join(processed_path, "thresh.png"), thresh)
        cv2.imwrite(os.path.join(processed_path, "edges.png"), edges)

        # Enhance contrast of original image
        pil_img = image.copy()
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(1.5)  # Boost contrast by 50%

        # Save enhanced version
        enhanced_path = os.path.join(processed_path, "enhanced.png")
        enhanced_img.save(enhanced_path)

        return {
            "original": image,
            "enhanced": enhanced_img,
            "paths": {
                "original": os.path.join(UPLOAD_FOLDER, "uploaded.png"),
                "enhanced": enhanced_path,
                "thresh": os.path.join(processed_path, "thresh.png"),
                "edges": os.path.join(processed_path, "edges.png"),
            },
        }
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        # Return original if processing fails
        return {
            "original": image,
            "enhanced": image,
            "paths": {"original": os.path.join(UPLOAD_FOLDER, "uploaded.png")},
        }


# Improved multimodal conversation handler
def handle_multimodal_conversation(
    message, processed_images, history, conversation_context
):
    """Handle conversation that involves both text and visual elements"""

    # Build conversation history for context
    context = ""
    if len(history) > 0:
        for msg in history[-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"

    # Determine if we need to include spatial context
    spatial_ref = ""
    if conversation_context.get("spatial_discussion", False):
        spatial_ref = "\nRemember to reference spatial elements by their position and relationship to other elements in the image."

    # Handle deictic references (this, that, there)
    deictic_handling = ""
    if conversation_context.get("deictic_reference", False):
        deictic_handling = "\nThe user is referencing specific elements in the image. Focus your response on visual elements that stand out in the design."

    # Prepare the prompt with enhanced spatial awareness
    prompt = f"""
{context}

User: {message}

{spatial_ref}
{deictic_handling}
"""


# Continuous voice listening mode
class VoiceListener:
    def __init__(self):
        self.is_listening = False
        self.listener_thread = None
        self.wake_words = ["hey spatial", "hey assistant", "spatial assistant"]
        self.timeout = 5  # seconds to wait for speech after wake word
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.callback = None  # Function to call when speech is detected

    def start_listening(self, callback_function):
        """Start listening for wake words in the background"""
        if self.is_listening:
            return False

        self.callback = callback_function
        self.is_listening = True
        self.listener_thread = threading.Thread(target=self._listen_loop)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        return True

    def stop_listening(self):
        """Stop the background listening"""
        self.is_listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
            self.listener_thread = None

    def _listen_loop(self):
        """Background thread that listens for wake words"""
        logger.info("Voice listener started")

        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    logger.debug("Listening for wake word...")

                    # Shorter listen for wake word detection
                    audio = self.recognizer.listen(
                        source, timeout=1, phrase_time_limit=2
                    )

                    try:
                        # Use Google for wake word detection (more accurate)
                        text = self.recognizer.recognize_google(audio).lower()
                        logger.debug(f"Heard: {text}")

                        # Check for wake words
                        if any(wake_word in text for wake_word in self.wake_words):
                            logger.info(f"Wake word detected: {text}")

                            # Acknowledge wake word
                            speak_text("Yes?", interrupt=True)

                            # Listen for command with longer timeout
                            try:
                                logger.info("Listening for command...")
                                audio = self.recognizer.listen(
                                    source, timeout=self.timeout, phrase_time_limit=10
                                )

                                # Process command with selected recognition engine
                                command = transcribe_audio(None, audio)

                                if command and command.strip():
                                    logger.info(f"Command: {command}")
                                    if self.callback:
                                        self.callback(command)
                                else:
                                    speak_text("I didn't catch that. Please try again.")
                            except sr.WaitTimeoutError:
                                speak_text("I'm still here if you need me.")

                    except sr.UnknownValueError:
                        # No speech detected, continue listening
                        pass
                    except sr.RequestError:
                        # Network error with Google, continue listening
                        pass
                    except Exception as e:
                        logger.error(f"Error in wake word detection: {e}")

            except Exception as e:
                logger.error(f"Error in listener loop: {e}")
                time.sleep(1)  # Prevent tight loop on error


# Updated transcribe_audio to handle direct audio data and files
def transcribe_audio(audio_path=None, audio_data=None):
    """
    Transcribe audio from either a file path or direct audio data
    """
    if not audio_path and not audio_data:
        logger.warning("No audio source provided for transcription")
        return ""

    try:
        logger.info(f"Transcribing audio {'from file' if audio_path else 'from data'}")

        # Prepare for recognition
        if audio_path:
            # Check if the audio file exists and has content
            if not os.path.exists(audio_path):
                logger.error(f"Audio file does not exist: {audio_path}")
                return (
                    "The audio file was not properly saved. Please try speaking again."
                )

            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                logger.error(f"Audio file is empty (0 bytes): {audio_path}")
                return "The audio file is empty. Please check your microphone permissions and try again."

            logger.info(f"Audio file size: {file_size} bytes")

            # Get audio data from file
            with sr.AudioFile(audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)

        # Ensure we have audio data at this point
        if not audio_data:
            return "No audio data to process"

        # Get audio duration
        audio_duration = len(audio_data.frame_data) / (
            audio_data.sample_rate * audio_data.sample_width
        )
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")

        if audio_duration < 0.5:
            logger.warning(f"Audio is very short: {audio_duration:.2f} seconds")
            return (
                "The recording was too short. Please speak for at least half a second."
            )

        # Determine which recognition engine to use (same as before)
        engine_to_use = RECOGNITION_ENGINE
        if engine_to_use == "auto":
            if WHISPER_AVAILABLE:
                engine_to_use = "whisper"
            elif VOSK_AVAILABLE:
                engine_to_use = "vosk"
            else:
                engine_to_use = "google"

        logger.info(f"Using {engine_to_use} for speech recognition")

        # The rest of the function remains the same as your original code,
        # except when using Whisper with direct audio data:
        if engine_to_use == "whisper" and WHISPER_AVAILABLE:
            if not audio_path:
                # Save audio data to temporary file for Whisper
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_filename = temp_file.name
                temp_file.close()

                import scipy.io.wavfile as wav

                wav.write(
                    temp_filename,
                    audio_data.sample_rate,
                    np.frombuffer(audio_data.frame_data, np.int16),
                )
                audio_path = temp_filename

            # Process with Whisper
            try:
                result = WHISPER_MODEL.transcribe(audio_path)
                text = result["text"].strip()

                # Clean up temp file if we created one
                if temp_filename:
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass

                logger.info(f"Whisper transcribed: '{text}'")
                return (
                    text if text else "I couldn't detect any speech. Please try again."
                )
            except Exception as e:
                logger.error(f"Whisper recognition error: {e}")
                # Fall back to next option
                if VOSK_AVAILABLE:
                    engine_to_use = "vosk"
                else:
                    engine_to_use = "google"

        # 2. Vosk (offline recognition)
        if engine_to_use == "vosk" and VOSK_AVAILABLE:
            try:
                import wave

                wf = wave.open(audio_path, "rb")
                # Check if the audio format is compatible with Vosk
                if (
                    wf.getnchannels() != 1
                    or wf.getsampwidth() != 2
                    or wf.getcomptype() != "NONE"
                ):
                    # Convert to required format if needed
                    logger.info("Converting audio to Vosk-compatible format")
                    # This is a simple conversion - in production you might want a more robust solution
                    temp_file = os.path.join(UPLOAD_FOLDER, "temp_mono.wav")
                    import subprocess

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            audio_path,
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            temp_file,
                        ],
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                    )
                    wf = wave.open(temp_file, "rb")

                rec = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
                rec.SetWords(True)

                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        results.append(json.loads(rec.Result())["text"])

                results.append(json.loads(rec.FinalResult())["text"])
                text = " ".join(results).strip()
                logger.info(f"Vosk transcribed: '{text}'")
                return (
                    text if text else "I couldn't detect any speech. Please try again."
                )
            except Exception as e:
                logger.error(f"Vosk recognition error: {e}")
                # Fall back to Google
                engine_to_use = "google"

        # 3. Google Speech Recognition (online)

        # Google Speech Recognition (with added logic for direct audio data)
        if engine_to_use == "google":
            try:
                logger.info(
                    f"Recognizing speech using Google (language: {SPEECH_LANGUAGE})..."
                )
                text = recognizer.recognize_google(audio_data, language=SPEECH_LANGUAGE)
                logger.info(f"Google transcribed: '{text}'")
                return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                return (
                    "I couldn't understand what you said. Could you try speaking again?"
                )
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition service error: {e}")
                return (
                    "I'm having trouble connecting to the speech recognition service."
                )
            except Exception as e:
                logger.error(f"Google recognition error: {e}")
                return "There was an issue with voice recognition."

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        traceback.print_exc()
        return "There was an issue with voice recognition. Please try typing your question."


# Voice agent command handler
def process_voice_command(command, image_input, history_state):
    """Process voice commands with special handling for control commands"""

    # Guard clause: ensure command is a string and not None
    if not isinstance(command, str) or command is None:
        logger.warning("Received invalid command input: %s", command)
        return history_state, "", history_state

    # If command is a file path (audio file), transcribe it first
    if command.endswith(".wav") or command.endswith(".mp3") or command.endswith(".m4a"):
        transcribed_text = transcribe_audio(audio_path=command)
        if not transcribed_text or transcribed_text.strip() == "":
            speak_text("I couldn't understand the audio. Please try again.")
            return history_state, "", history_state
        command = transcribed_text

    # Check for control commands
    if any(x in command.lower() for x in ["quit", "exit", "stop", "end"]):
        speak_text("Goodbye! Feel free to talk to me again when you need help.")
        return history_state, "", history_state

    if "clear" in command.lower() and any(
        x in command.lower() for x in ["screen", "chat", "history", "conversation"]
    ):
        speak_text("Clearing our conversation.")
        return [], "", []

    if any(
        x in command.lower()
        for x in ["take picture", "take a picture", "capture", "snapshot"]
    ):
        speak_text(
            "I don't have camera access. Please upload an image using the interface."
        )
        notification = (
            "To analyze an image, please upload it using the image upload area."
        )
        history_state.append({"role": "assistant", "content": notification})
        return history_state, "", history_state

    # Process as normal message if not a control command
    return chatbot_interface(image_input, command, history_state)


# Add continuous listening toggle to UI
def toggle_voice_listener(active, history_state):
    """Toggle continuous voice listening mode"""
    global voice_listener

    if active:
        # Initialize listener if needed
        if not hasattr(globals(), "voice_listener") or voice_listener is None:
            voice_listener = VoiceListener()

        # Define callback to handle commands
        def voice_command_callback(command):
            nonlocal history_state
            history_state, _, history_state = process_voice_command(
                command, None, history_state
            )
            # Update UI - note: this requires Gradio's live updates feature
            chatbot.update(history_state)

        # Start listening
        if voice_listener.start_listening(voice_command_callback):
            notification = (
                "Voice assistant activated. Say 'Hey Spatial' to get my attention."
            )
            speak_text(notification)
            history_state.append({"role": "assistant", "content": notification})
            return history_state
        else:
            notification = "Voice assistant is already running."
            history_state.append({"role": "assistant", "content": notification})
            return history_state
    else:
        # Stop listening
        if hasattr(globals(), "voice_listener") and voice_listener:
            voice_listener.stop_listening()
            notification = "Voice assistant deactivated."
            speak_text(notification)
            history_state.append({"role": "assistant", "content": notification})
            return history_state
        return history_state


# Save chat history
def download_chat(history) -> str:
    path = os.path.join(UPLOAD_FOLDER, "chat_log.txt")
    with open(path, "w") as f:
        for msg in history:
            f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
    return path


# Clean up function for when the app exits
def on_close():
    global tts_thread_active
    tts_thread_active = False
    logger.info("Shutting down TTS thread...")
    pygame.mixer.quit()


# Initialize history with welcome message
def initialize_chat(history=[]):
    if not history:
        welcome = get_welcome_message()
        history = [{"role": "assistant", "content": welcome}]
        global conversation_context
        conversation_context["greeting_done"] = True
        speak_text(welcome)
    return history


# Reset conversation context when starting a new conversation
def clear_conversation():
    global conversation_context
    global image_desc

    # Reset conversation state
    conversation_context = {
        "current_topic": None,
        "last_question": None,
        "greeting_done": False,
        "image_uploaded": False,
        "image_analyzed": False,
    }

    # Reset image description
    image_desc = None

    # Return empty state for UI components
    return [], "", []


# Test microphone function to check if it's working
def test_microphone():
    try:
        with sr.Microphone() as source:
            logger.info("Microphone test: Device initialized")
            audio = recognizer.listen(source, timeout=3)
            logger.info("Microphone test: Audio captured successfully")
            return "Microphone is working properly! I can hear you."
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        return f"Microphone test failed: {str(e)}"


# Enhanced UI with Voice Agent Mode
demo = gr.Blocks(
    css="""
.conversation-container {
    border-radius: 10px;
    background-color: #f8f9fa;
    padding: 20px;
}
.voice-container {
    display: flex;
    align-items: center;
    gap: 10px;
}
.header {
    text-align: center;
    margin-bottom: 20px;
}
.mic-status {
    font-size: 14px;
    color: #666;
    margin-top: 5px;
}
.voice-mode-active {
    background-color: #d4edda !important;
    border-color: #c3e6cb !important;
}
.voice-mode-inactive {
    background-color: #f8d7da !important;
    border-color: #f5c6cb !important;
}
.status-light {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 5px;
}
.status-light.active {
    background-color: #28a745;
    box-shadow: 0 0 10px #28a745;
}
.status-light.inactive {
    background-color: #dc3545;
}
"""
)

with demo:
    # State variables
    history_state = gr.State([])
    voice_mode_state = gr.State(False)

    gr.Markdown("## 🧠 Spatial Reasoning Voice Assistant", elem_classes=["header"])

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload your sketch (JPG/PNG)",
                height=450,
                width="100%",
            )

            # Voice agent controls section
            with gr.Group(elem_classes=["voice-controls"]):
                gr.Markdown("### 🎙️ Voice Assistant Controls")

                with gr.Row():
                    voice_toggle = gr.Checkbox(
                        label="Activate Voice Assistant",
                        value=False,
                        info="Toggle always-listening mode",
                    )

                    # Display current status with indicator light
                    voice_status = gr.Markdown(
                        '<span class="status-light inactive"></span> Voice assistant is inactive',
                        elem_classes=["voice-status-indicator"],
                    )

                with gr.Row():
                    wake_word_text = gr.Textbox(
                        label="Wake Word",
                        value="Hey Spatial",
                        info="The phrase to activate the assistant (in always-listening mode)",
                    )

                    tts_voice_selector = gr.Dropdown(
                        label="Assistant Voice",
                        choices=[
                            "Default Female",
                            "Default Male",
                            "Premium (ElevenLabs)",
                        ],
                        value="Default Female",
                        info="Select the voice for text-to-speech",
                    )

                gr.Markdown(
                    """
                **Voice Commands:**
                - "Hey Spatial, analyze this sketch"
                - "Hey Spatial, what do you see in this image?"
                - "Hey Spatial, clear the conversation"
                """
                )

            with gr.Row():
                download_btn = gr.Button("💾 Save Conversation")
                clear_btn = gr.Button("🧼 New Conversation")
                test_mic_btn = gr.Button("🔊 Test Microphone")

            # Speech recognition engine selector
            speech_engine = gr.Radio(
                choices=["Auto", "Google", "Vosk (offline)", "Whisper (offline)"],
                value="Auto",
                label="Speech Recognition Engine",
                info="Choose which speech recognition system to use",
            )

        with gr.Column(scale=2, elem_classes=["conversation-container"]):
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=450,
                show_label=False,
                elem_id="chatbox",
                render=True,  # Important for real-time updates
            )
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Type your message or ask a question...",
                    show_label=False,
                    scale=5,
                )
                audio_input = gr.Microphone(
                    label="🎙️",
                    type="filepath",
                    interactive=True,
                    streaming=False,
                    scale=1,
                )

            mic_status = gr.Textbox(
                label="Voice Recognition Status",
                value="Click 'Test Microphone' to check if your microphone is working",
                interactive=False,
                elem_classes=["mic-status"],
            )

    # Initialize chat with welcome message
    demo.load(initialize_chat, [], chatbot)

    # Toggle continuous voice listening mode
    voice_toggle.change(
        toggle_voice_listener, [voice_toggle, history_state], [history_state]
    )

    # Update voice status indicator styling
    voice_toggle.change(
        lambda active: f'<span class="status-light {"active" if active else "inactive"}"></span> Voice assistant is {"active" if active else "inactive"}',
        [voice_toggle],
        [voice_status],
    )

    # Handle voice assistant parameters
    def update_voice_assistant_params(wake_word, voice_type):
        global voice_listener, tts_voice_type

        # Update wake word if voice listener exists
        if hasattr(globals(), "voice_listener") and voice_listener:
            # Extract wake word and add variations
            base_wake = wake_word.lower().strip()
            voice_listener.wake_words = [
                base_wake,
                f"hey {base_wake}",
                f"{base_wake} assistant",
            ]

        # Update TTS voice
        voice_map = {
            "Default Female": {"engine": "pyttsx3", "voice_id": 1},
            "Default Male": {"engine": "pyttsx3", "voice_id": 0},
            "Premium (ElevenLabs)": {"engine": "elevenlabs", "voice_id": "default"},
        }

        tts_voice_type = voice_map.get(
            voice_type, {"engine": "gtts", "voice_id": "default"}
        )

        # Apply voice setting if using pyttsx3
        try:
            if tts_voice_type["engine"] == "pyttsx3" and hasattr(
                globals(), "tts_engine"
            ):
                voices = tts_engine.getProperty("voices")
                if len(voices) > tts_voice_type["voice_id"]:
                    tts_engine.setProperty(
                        "voice", voices[tts_voice_type["voice_id"]].id
                    )
        except Exception as e:
            logger.error(f"Error setting voice: {e}")

        return f"Updated: Wake word set to '{wake_word}' and voice to '{voice_type}'"

    # Connect wake word and voice selection
    wake_word_text.change(
        update_voice_assistant_params,
        [wake_word_text, tts_voice_selector],
        [mic_status],
    )

    tts_voice_selector.change(
        update_voice_assistant_params,
        [wake_word_text, tts_voice_selector],
        [mic_status],
    )

    # Original connections
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
        process_voice_command,
        [audio_input, image_input, history_state],
        [chatbot, text_input, history_state],
    )
    clear_btn.click(
        lambda: ([], "", []),
        [],
        [chatbot, text_input, history_state],
    )
    download_btn.click(download_chat, [history_state], gr.File())
    test_mic_btn.click(test_microphone, [], mic_status)

    # Connect speech engine selection to recognition engine update
    def update_speech_engine(choice):
        global RECOGNITION_ENGINE
        if choice == "Auto":
            RECOGNITION_ENGINE = "auto"
        elif choice == "Google":
            RECOGNITION_ENGINE = "google"
        elif choice == "Vosk (offline)":
            RECOGNITION_ENGINE = "vosk"
        elif choice == "Whisper (offline)":
            RECOGNITION_ENGINE = "whisper"

        # Provide feedback about available engines
        if RECOGNITION_ENGINE == "vosk" and not VOSK_AVAILABLE:
            return "Vosk is not installed or model is missing. Please install vosk and download a model."
        elif RECOGNITION_ENGINE == "whisper" and not WHISPER_AVAILABLE:
            return "Whisper is not installed. Please install openai-whisper to use this engine."
        else:
            return f"Speech recognition set to: {choice}"

    speech_engine.change(update_speech_engine, speech_engine, mic_status)

    # Register cleanup function
    atexit.register(on_close)

if __name__ == "__main__":
    start_tts_thread()  # Make sure TTS is ready
    logger.info("Starting Spatial Reasoning Conversation Assistant...")

    # Check and display the availability of speech recognition engines
    available_engines = ["Google API (online)"]
    if VOSK_AVAILABLE:
        available_engines.append("Vosk (offline)")
    if WHISPER_AVAILABLE:
        available_engines.append("Whisper (offline)")

    logger.info(f"Available speech recognition engines: {', '.join(available_engines)}")

    # Launch the app
    demo.launch(share=True)
