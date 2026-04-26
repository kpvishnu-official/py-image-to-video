"""
generate_final_video.py
=======================
Audiobook Video Generator — Main Entry Point

Usage:
    python generate_final_video.py test_story.txt

What it does:
    1. Reads and splits the story text into chunks
    2. Generates a TTS audio clip for each chunk (gTTS)
    3. Generates an image for each chunk (Stable Diffusion)
    4. Assembles everything into a video with crossfade transitions (MoviePy)
    5. Saves the final video as output/audiobook.mp4

Requirements (install once):
    pip install gtts diffusers transformers accelerate torch pillow moviepy

System requirement:
    ffmpeg must be installed:
      - Ubuntu/Debian : sudo apt install ffmpeg
      - macOS         : brew install ffmpeg
      - Windows       : https://ffmpeg.org/download.html
"""

import sys
import os
import time

from text_processor import TextProcessor
from audio_generator import AudioGenerator
from image_generator import ImageGenerator
from video_assembler import VideoAssembler


# ------------------------------------------------------------------ #
#  Configuration — edit these to customise the output                  #
# ------------------------------------------------------------------ #

OUTPUT_DIR          = "output"               # Root folder for all generated files
AUDIO_DIR           = f"{OUTPUT_DIR}/audio"  # Where .mp3 files are saved
IMAGE_DIR           = f"{OUTPUT_DIR}/images" # Where .png files are saved
FINAL_VIDEO_PATH    = f"{OUTPUT_DIR}/audiobook.mp4"

CHUNK_SIZE          = 300    # Characters per text chunk (smaller = more scenes)
TTS_LANGUAGE        = "en"   # Language for TTS: 'en', 'hi', 'fr', etc.
VIDEO_FPS           = 24     # Frames per second
TRANSITION_SECONDS  = 1.0    # Crossfade duration between images (seconds)


# ------------------------------------------------------------------ #
#  Main pipeline                                                        #
# ------------------------------------------------------------------ #

def main():
    # ── 0. Parse command-line argument ──────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python generate_final_video.py <story_file.txt>")
        sys.exit(1)

    story_file = sys.argv[1]

    print("=" * 60)
    print("  Audiobook Video Generator")
    print("=" * 60)
    print(f"  Story file : {story_file}")
    print(f"  Output     : {FINAL_VIDEO_PATH}")
    print("=" * 60)

    start_time = time.time()

    # ── 1. Text processing ──────────────────────────────────────────
    print("\n[Step 1/4] Loading and splitting text ...")
    text_processor = TextProcessor(
        file_path=story_file,
        chunk_size=CHUNK_SIZE,
    )
    text_processor.load_and_split()
    chunks = text_processor.get_chunks()
    print(f"  → {len(chunks)} chunks ready.\n")

    # ── 2. Load Stable Diffusion model ──────────────────────────────
    print("[Step 2/4] Loading Stable Diffusion model ...")
    print("  (First run downloads ~4 GB — subsequent runs load from cache)\n")
    image_generator = ImageGenerator(output_dir=IMAGE_DIR)
    image_generator.load_model()

    # ── 3. Initialise the other components ──────────────────────────
    audio_generator = AudioGenerator(
        output_dir=OUTPUT_DIR + "/audio",
        language=TTS_LANGUAGE,
    )
    video_assembler = VideoAssembler(
        output_path=FINAL_VIDEO_PATH,
        fps=VIDEO_FPS,
        transition_duration=TRANSITION_SECONDS,
    )

    # ── 4. Process each chunk ────────────────────────────────────────
    print(f"\n[Step 3/4] Processing {len(chunks)} chunks ...")
    for i, chunk in enumerate(chunks):
        print(f"\n  ── Chunk {i + 1}/{len(chunks)} ──────────────────────────")
        print(f"  Text preview: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")

        # Generate audio
        audio_path = audio_generator.generate(text=chunk, index=i)
        duration   = audio_generator.get_duration(audio_path)
        print(f"  Audio duration: {duration:.2f}s")

        # Generate image
        image_path = image_generator.generate(text=chunk, index=i)

        # Queue this segment
        video_assembler.add_segment(
            image_path=image_path,
            audio_path=audio_path,
            duration=duration,
        )

    # ── 5. Assemble and export ───────────────────────────────────────
    print(f"\n[Step 4/4] Assembling final video ...")
    video_path = video_assembler.assemble()

    elapsed = time.time() - start_time
    minutes  = int(elapsed // 60)
    seconds  = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("  ✓ Done!")
    print(f"  Output video : {os.path.abspath(video_path)}")
    print(f"  Total time   : {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
