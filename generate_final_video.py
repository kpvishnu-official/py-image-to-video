"""
generate_final_video.py
=======================
Audiobook Video Generator -- Main Entry Point

Usage:
    python generate_final_video.py test_story.txt
    python generate_final_video.py test_story.txt --music background.mp3

What it does:
    1. Reads and splits the story text into chunks
    2. Generates natural-sounding narration audio (Coqui TTS -- offline)
    3. Optionally mixes soft background music under the narration
    4. Generates an image for each chunk (Stable Diffusion)
    5. Assembles everything into a video with crossfade transitions (MoviePy)
    6. Saves the final video as output/audiobook.mp4

Install dependencies (run once):
    pip install TTS pydub diffusers transformers accelerate torch pillow moviepy
    sudo apt install ffmpeg        # Ubuntu/Debian
    brew install ffmpeg            # macOS

TTS voices you can try (edit TTS_MODEL below):
    "tts_models/en/jenny/jenny"                     -- warm female, audiobook style (default)
    "tts_models/en/ljspeech/glow-tts"              -- clear female, fast
    "tts_models/en/vctk/vits"                      -- multiple accents/speakers
    "tts_models/multilingual/multi-dataset/xtts_v2" -- highest quality, slowest
"""

import sys
import os
import time
import argparse

from text_processor  import TextProcessor
from audio_generator import AudioGenerator
from image_generator import ImageGenerator
from video_assembler import VideoAssembler


# ------------------------------------------------------------------ #
#  Configuration -- edit these to customise the output                 #
# ------------------------------------------------------------------ #

OUTPUT_DIR         = "output"
AUDIO_DIR          = f"{OUTPUT_DIR}/audio"
IMAGE_DIR          = f"{OUTPUT_DIR}/images"
FINAL_VIDEO_PATH   = f"{OUTPUT_DIR}/audiobook.mp4"

CHUNK_SIZE         = 300    # Characters per chunk (smaller = more scenes)

# TTS voice model -- see docstring at top for alternatives
TTS_MODEL          = "tts_models/en/jenny/jenny"

# Background music volume (0.0 = off, 0.10 = 10% = subtle, 0.25 = noticeable)
BG_MUSIC_VOLUME    = 0.10

VIDEO_FPS          = 24
TRANSITION_SECONDS = 1.0


# ------------------------------------------------------------------ #
#  Argument parsing                                                     #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an audiobook video from a .txt story file."
    )
    parser.add_argument(
        "story_file",
        help="Path to your story .txt file"
    )
    parser.add_argument(
        "--music",
        default=None,
        metavar="PATH",
        help="(Optional) Path to a background music .mp3 or .wav file"
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=BG_MUSIC_VOLUME,
        metavar="0.0-1.0",
        help=f"Volume of background music (default: {BG_MUSIC_VOLUME})"
    )
    parser.add_argument(
        "--voice",
        default=TTS_MODEL,
        metavar="MODEL",
        help=f"Coqui TTS model name (default: {TTS_MODEL})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        metavar="N",
        help=f"Characters per scene chunk (default: {CHUNK_SIZE})"
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main pipeline                                                        #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    print("=" * 60)
    print("  Audiobook Video Generator")
    print("=" * 60)
    print(f"  Story file  : {args.story_file}")
    print(f"  Voice model : {args.voice}")
    print(f"  Background  : {args.music or 'none'}")
    print(f"  Output      : {FINAL_VIDEO_PATH}")
    print("=" * 60)

    start_time = time.time()

    # ── 1. Text processing ──────────────────────────────────────────
    print("\n[Step 1/4] Loading and splitting text ...")
    text_processor = TextProcessor(
        file_path=args.story_file,
        chunk_size=args.chunk_size,
    )
    text_processor.load_and_split()
    chunks = text_processor.get_chunks()
    print(f"  -> {len(chunks)} chunks ready.\n")

    # ── 2. Load models ──────────────────────────────────────────────
    print("[Step 2/4] Loading AI models ...")

    audio_generator = AudioGenerator(
        output_dir=AUDIO_DIR,
        model_name=args.voice,
        bg_music_path=args.music,
        bg_music_volume=args.music_volume,
    )
    audio_generator.load_model()

    print()
    image_generator = ImageGenerator(output_dir=IMAGE_DIR)
    image_generator.load_model()

    video_assembler = VideoAssembler(
        output_path=FINAL_VIDEO_PATH,
        fps=VIDEO_FPS,
        transition_duration=TRANSITION_SECONDS,
    )

    # ── 3. Process each chunk ────────────────────────────────────────
    print(f"\n[Step 3/4] Processing {len(chunks)} chunks ...")

    for i, chunk in enumerate(chunks):
        print(f"\n  -- Chunk {i + 1}/{len(chunks)} " + "-" * 30)
        print(f"  Text: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")

        audio_path = audio_generator.generate(text=chunk, index=i)
        duration   = audio_generator.get_duration(audio_path)
        print(f"  Duration: {duration:.2f}s")

        image_path = image_generator.generate(text=chunk, index=i)

        video_assembler.add_segment(
            image_path=image_path,
            audio_path=audio_path,
            duration=duration,
        )

    # ── 4. Assemble and export ───────────────────────────────────────
    print(f"\n[Step 4/4] Assembling final video ...")
    video_path = video_assembler.assemble()

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Output video : {os.path.abspath(video_path)}")
    print(f"  Total time   : {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
