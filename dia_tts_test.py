import logging

try:
    from dia.model import Dia

    DIA_AVAILABLE = True
except ImportError:
    DIA_AVAILABLE = False
    logging.error("Dia TTS model not installed or not found.")


def test_dia_tts():
    if not DIA_AVAILABLE:
        print("Dia TTS model not available.")
        return
    try:
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
        print("Dia TTS model loaded successfully.")
        text = "[S1] Hello, this is a test of the Dia TTS model."
        output = model.generate(text, use_torch_compile=False, verbose=False)
        temp_file = "test_output.wav"
        model.save_audio(temp_file, output)
        print(f"Audio saved to {temp_file}. Please play it to verify.")
    except Exception as e:
        print(f"Error during Dia TTS test: {e}")


if __name__ == "__main__":
    test_dia_tts()
