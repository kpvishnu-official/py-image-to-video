import os
import torch
from TTS.api import TTS
from pydub import AudioSegment
from moviepy.editor import AudioFileClip

class AudioGenerator:
    """
    Converts text chunks into high-quality narration audio using Coqui TTS.
    Optionally mixes in soft background music underneath the narration.

    Install:
        pip install TTS pydub
        sudo apt install ffmpeg   (or brew install ffmpeg on Mac)

    Available models (pass as model_name):
        "tts_models/en/ljspeech/tacotron2-DDC"          -- clear female, fast
        "tts_models/en/ljspeech/glow-tts"               -- clear female, natural
        "tts_models/en/vctk/vits"                       -- multi-speaker, many accents
        "tts_models/en/jenny/jenny"                     -- warm female, audiobook style
        "tts_models/multilingual/multi-dataset/xtts_v2" -- best quality, slowest

    For audiobooks, "tts_models/en/jenny/jenny" is recommended.
    First run downloads the model (~few hundred MB, cached after that).
    """

    DEFAULT_MODEL = "tts_models/en/jenny/jenny"

    def __init__(
        self,
        output_dir: str = "output/audio",
        model_name: str = DEFAULT_MODEL,
        bg_music_path: str = None,
        bg_music_volume: float = 0.10,
    ):
        """
        Args:
            output_dir       : Folder where audio files will be saved.
            model_name       : Coqui TTS model to use. See class docstring.
            bg_music_path    : Path to a background music .mp3/.wav file.
                               Set to None to disable background music.
            bg_music_volume  : Volume of background music (0.0 to 1.0).
                               0.10 = 10% -- subtle, sitting under the voice.
        """
        self.output_dir      = output_dir
        self.model_name      = model_name
        self.bg_music_path   = bg_music_path
        self.bg_music_volume = bg_music_volume
        self.tts             = None   # loaded via load_model()

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public methods                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """
        Downloads (first time) and loads the Coqui TTS model into memory.
        Call this once before looping over generate().
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [AudioGenerator] Loading TTS model '{self.model_name}' on {device.upper()} ...")
        print("  [AudioGenerator] First run downloads the model -- please wait ...")

        self.tts = TTS(model_name=self.model_name, progress_bar=True).to(device)
        print("  [AudioGenerator] TTS model ready.")

        if self.bg_music_path:
            if not os.path.exists(self.bg_music_path):
                print(f"  [AudioGenerator] WARNING: Music file not found at "
                      f"'{self.bg_music_path}'. Background music will be skipped.")
                self.bg_music_path = None
            else:
                print(f"  [AudioGenerator] Background music: {self.bg_music_path}")

    def generate(self, text: str, index: int) -> str:
        """
        Generates a narration audio file for a text chunk.
        If bg_music_path was set, mixes soft background music underneath.

        Args:
            text  : The story text to narrate.
            index : Chunk number -- used for naming the output file.

        Returns:
            Path to the final .wav file.
        """
        if self.tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        final_path = os.path.join(self.output_dir, f"audio_{index:04d}.wav")

        # Skip re-generation on repeated runs (saves a lot of time)
        if os.path.exists(final_path):
            print(f"  [AudioGenerator] Reusing existing: {final_path}")
            return final_path

        # Step 1: Generate raw narration wav
        raw_path = os.path.join(self.output_dir, f"raw_{index:04d}.wav")
        print(f"  [AudioGenerator] Generating narration for chunk {index} ...")
        self.tts.tts_to_file(text=text, file_path=raw_path)

        # Step 2: Mix with background music if provided
        if self.bg_music_path:
            print("  [AudioGenerator] Mixing background music ...")
            self._mix_with_music(raw_path, final_path)
            os.remove(raw_path)
        else:
            os.rename(raw_path, final_path)

        print(f"  [AudioGenerator] Saved: {final_path}")
        return final_path

    def get_duration(self, audio_path: str) -> float:
        """Returns the duration of an audio file in seconds."""
        clip = AudioFileClip(audio_path)
        duration = clip.duration
        clip.close()
        return duration

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _mix_with_music(self, speech_path: str, output_path: str) -> None:
        """
        Overlays the narration on top of softly-looped background music.
        Music is automatically looped or trimmed to match narration length,
        with fade-in and fade-out applied for a polished sound.
        """
        speech = AudioSegment.from_wav(speech_path)
        music  = AudioSegment.from_file(self.bg_music_path)

        # Convert 0.0-1.0 volume to a dB adjustment
        # e.g. 0.10 -> ~-20 dB (very subtle under the voice)
        import math
        if self.bg_music_volume > 0:
            volume_db = 20 * math.log10(self.bg_music_volume)
        else:
            volume_db = -120  # effectively silent

        music = music + volume_db

        speech_ms = len(speech)

        # Loop music to be at least as long as speech
        if len(music) < speech_ms:
            repeats = (speech_ms // len(music)) + 2
            music   = music * repeats

        music = music[:speech_ms]

        # Smooth fade in/out on music
        fade_ms = min(2000, speech_ms // 4)
        music   = music.fade_in(fade_ms).fade_out(fade_ms)

        # Overlay: speech sits on top, music underneath
        mixed = music.overlay(speech)
        mixed.export(output_path, format="wav")
