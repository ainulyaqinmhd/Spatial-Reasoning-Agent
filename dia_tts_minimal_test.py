import time
import tempfile
import os
import pygame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dia.model import Dia
except ImportError:
    logger.error(
        "Dia TTS model not installed. Please install it with 'pip install git+https://github.com/nari-labs/dia.git'"
    )
    exit(1)


def test_dia_tts(text):
    try:
        pygame.mixer.init()
        logger.info("Pygame mixer initialized successfully")
    except Exception as e:
        logger.error(f"Pygame mixer initialization failed: {e}")
        return

    try:
        logger.info("Loading Dia TTS model...")
        dia_tts_model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B", compute_dtype="float16"
        )
        logger.info("Dia TTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Dia TTS model: {e}")
        return

    try:
        if not text.strip().startswith("[S1]") and not text.strip().startswith("[S2]"):
            text_to_speak = "[S1] " + text.strip()
        else:
            text_to_speak = text.strip()

        logger.info("Generating audio with Dia TTS...")
        output = dia_tts_model.generate(
            text_to_speak, use_torch_compile=False, verbose=False
        )
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.close()

        dia_tts_model.save_audio(temp_filename, output)
        logger.info(f"Audio saved to {temp_filename}")

        if os.path.exists(temp_filename):
            file_size = os.path.getsize(temp_filename)
            logger.info(f"Generated audio file size: {file_size} bytes")
            if file_size == 0:
                logger.error("Generated audio file is empty!")
                return
        else:
            logger.error("Generated audio file does not exist!")
            return

        logger.info("Playing audio...")
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        logger.info("Finished playing audio")

        try:
            os.unlink(temp_filename)
        except Exception as e:
            logger.warning(f"Failed to delete temp audio file: {e}")

    except Exception as e:
        logger.error(f"Error during Dia TTS generation or playback: {e}")


if __name__ == "__main__":
    test_text = """1. You usually start by navigating to the location where you want your project to live.
2. If the project directory doesn't exist, you create it.
3. Next, you create an isolated virtual environment for your project's dependencies.
4. You then activate this environment so that any packages you install are contained within it.
5. Opening the project in VS Code is a natural next step to start working on your code within the activated environment.
6. As you develop, you'll create new files to organize your project."""
    test_dia_tts(test_text)
