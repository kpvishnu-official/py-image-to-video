"""
generate_final_video.py
=======================
Audiobook Video Generator -- Main Entry Point

Usage:
    python generate_final_video.py test_story.txt
    python generate_final_video.py test_story.txt --music background.mp3

Pipeline:
    1. Read & split story text into chunks
    2. FIRST PASS: scan all chunks for characters, build visual profiles
    3. Generate natural TTS narration (Coqui TTS -- offline)
    4. Generate image per chunk with correct characters injected (SD)
    5. Assemble into video with crossfade transitions (MoviePy)
    6. Save as output/audiobook.mp4

Install:
    pip install TTS pydub diffusers transformers accelerate torch pillow moviepy
    sudo apt install ffmpeg
"""

import sys
import os
import time
import argparse

from text_processor    import TextProcessor
from audio_generator   import AudioGenerator
from image_generator   import ImageGenerator
from video_assembler   import VideoAssembler
from character_tracker import CharacterTracker


# ------------------------------------------------------------------ #
#  Configuration                                                        #
# ------------------------------------------------------------------ #

OUTPUT_DIR         = "output"
AUDIO_DIR          = f"{OUTPUT_DIR}/audio"
IMAGE_DIR          = f"{OUTPUT_DIR}/images"
FINAL_VIDEO_PATH   = f"{OUTPUT_DIR}/audiobook.mp4"

CHUNK_SIZE         = 300
TTS_MODEL          = "tts_models/en/jenny/jenny"
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
    parser.add_argument("story_file", help="Path to story .txt file")
    parser.add_argument("--music",        default=None,          metavar="PATH",
                        help="Background music .mp3 or .wav file")
    parser.add_argument("--music-volume", type=float, default=BG_MUSIC_VOLUME,
                        metavar="0.0-1.0", help="Background music volume")
    parser.add_argument("--voice",        default=TTS_MODEL,     metavar="MODEL",
                        help="Coqui TTS model name")
    parser.add_argument("--chunk-size",   type=int, default=CHUNK_SIZE,
                        metavar="N",      help="Characters per chunk")
    parser.add_argument("--show-characters", action="store_true",
                        help="Print full character report and exit (no video)")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main pipeline                                                        #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    print("=" * 60)
    print("  Audiobook Video Generator")
    print("=" * 60)
    print(f"  Story      : {args.story_file}")
    print(f"  Voice      : {args.voice}")
    print(f"  Music      : {args.music or 'none'}")
    print(f"  Output     : {FINAL_VIDEO_PATH}")
    print("=" * 60)

    start_time = time.time()

    # ── 1. Split text ────────────────────────────────────────────────
    print("\n[Step 1/5] Loading and splitting text ...")
    text_processor = TextProcessor(file_path=args.story_file,
                                   chunk_size=args.chunk_size)
    text_processor.load_and_split()
    chunks = text_processor.get_chunks()
    print(f"  -> {len(chunks)} chunks ready.")

    # ── 2. Character analysis (first pass) ───────────────────────────
    print("\n[Step 2/5] Analysing characters ...")
    tracker = CharacterTracker()
    tracker.analyse_all_chunks(chunks)
    tracker.print_character_report()

    # If user just wants the character report, stop here
    if args.show_characters:
        print("  --show-characters flag set. Exiting before generation.")
        return

    # ── 3. Load AI models ────────────────────────────────────────────
    print("\n[Step 3/5] Loading AI models ...")

    audio_generator = AudioGenerator(
        output_dir=AUDIO_DIR,
        model_name=args.voice,
        bg_music_path=args.music,
        bg_music_volume=args.music_volume,
    )
    audio_generator.load_model()

    image_generator = ImageGenerator(output_dir=IMAGE_DIR)
    image_generator.load_model()

    video_assembler = VideoAssembler(
        output_path=FINAL_VIDEO_PATH,
        fps=VIDEO_FPS,
        transition_duration=TRANSITION_SECONDS,
    )

    # ── 4. Process each chunk ────────────────────────────────────────
    print(f"\n[Step 4/5] Processing {len(chunks)} chunks ...")

    for i, chunk in enumerate(chunks):
        print(f"\n  -- Chunk {i+1}/{len(chunks)} " + "-" * 30)
        print(f"  Text: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")

        # Get character anchor for this specific chunk
        character_anchor = tracker.get_scene_anchor(chunk, i)
        print(f"  Characters in scene: {character_anchor[:80]}...")

        # Audio
        audio_path = audio_generator.generate(text=chunk, index=i)
        duration   = audio_generator.get_duration(audio_path)
        print(f"  Audio duration: {duration:.2f}s")

        # Image -- character anchor injected here
        image_path = image_generator.generate(
            text=chunk,
            index=i,
            character_anchor=character_anchor,
        )

        video_assembler.add_segment(
            image_path=image_path,
            audio_path=audio_path,
            duration=duration,
        )

    # ── 5. Assemble video ────────────────────────────────────────────
    print(f"\n[Step 5/5] Assembling final video ...")
    video_path = video_assembler.assemble()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Output : {os.path.abspath(video_path)}")
    print(f"  Time   : {int(elapsed//60)}m {int(elapsed%60)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
