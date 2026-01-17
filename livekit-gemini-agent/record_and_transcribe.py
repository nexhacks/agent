import argparse
import json
import os
import sys
import wave

import sounddevice as sd


def list_devices() -> None:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"[{idx}] {dev['name']} (in={dev['max_input_channels']})")


def record_audio(seconds: float, sample_rate: int, device: int | None) -> bytes:
    print(f"Recording {seconds:.1f}s at {sample_rate}Hz...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return audio.tobytes()


def write_wav(path: str, audio_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


def transcribe_vosk(path: str, model_path: str, sample_rate: int) -> None:
    try:
        import vosk
    except ImportError as exc:
        raise RuntimeError("Vosk not installed. Use --engine whisper on macOS.") from exc

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Vosk model not found at {model_path}. Download a model and set --vosk-model."
        )

    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, sample_rate)

    last_partial = ""
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("WAV must be mono 16-bit PCM.")
        if wf.getframerate() != sample_rate:
            raise ValueError("WAV sample rate does not match --sample-rate.")

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result()).get("text", "")
                if result:
                    print(f"[final] {result}")
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial and partial != last_partial:
                    print(f"[partial] {partial}")
                    last_partial = partial

        final = json.loads(rec.FinalResult()).get("text", "")
        if final:
            print(f"[final] {final}")


def transcribe_whisper(path: str, model_name: str) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper not installed. Use --engine vosk on Linux.") from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path)
    print(f"[info] language={info.language} probability={info.language_probability:.2f}")
    for segment in segments:
        text = segment.text.strip()
        if text:
            print(f"[{segment.start:.2f}-{segment.end:.2f}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record mic audio and transcribe with Vosk.")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "vosk", "whisper"],
        default="auto",
        help="Speech-to-text engine (auto picks whisper on macOS).",
    )
    parser.add_argument(
        "--vosk-model",
        type=str,
        default="models/vosk-model-small-en-us-0.15",
        help="Path to a Vosk model directory.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model name for faster-whisper.",
    )
    parser.add_argument("--wav", type=str, default="recording.wav")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    audio_bytes = record_audio(args.seconds, args.sample_rate, args.device)
    write_wav(args.wav, audio_bytes, args.sample_rate)
    print(f"Wrote {args.wav}")
    engine = args.engine
    if engine == "auto":
        engine = "whisper" if sys.platform == "darwin" else "vosk"

    if engine == "whisper":
        transcribe_whisper(args.wav, args.whisper_model)
    else:
        transcribe_vosk(args.wav, args.vosk_model, args.sample_rate)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
