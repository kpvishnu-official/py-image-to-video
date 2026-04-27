import os

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


class VideoAssembler:
    """
    Assembles image clips and audio clips into a final video with smooth
    crossfade transitions between each scene.

    Each segment = one image shown for the duration of its audio clip.
    Segments are concatenated with a short crossfade between them.
    """

    def __init__(
        self,
        output_path: str = "output/audiobook.mp4",
        fps: int = 24,
        transition_duration: float = 1.0,
    ):
        """
        Args:
            output_path         : Path where the final .mp4 will be saved.
            fps                 : Frames per second for the video. 24 is standard.
            transition_duration : Crossfade length in seconds between images.
        """
        self.output_path = output_path
        self.fps = fps
        self.transition_duration = transition_duration

        # Each entry is a dict: {"image": path, "audio": path, "duration": seconds}
        self._segments: list[dict] = []

        # Make sure the output folder exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public methods                                                       #
    # ------------------------------------------------------------------ #

    def add_segment(self, image_path: str, audio_path: str, duration: float) -> None:
        """
        Registers one scene (image + audio pair) to be included in the video.

        Args:
            image_path : Path to the .png image file for this scene.
            audio_path : Path to the .mp3 audio file for this scene.
            duration   : How long (seconds) this scene should stay on screen.
                         Should match the audio duration.
        """
        self._segments.append({
            "image": image_path,
            "audio": audio_path,
            "duration": duration,
        })
        print(f"  [VideoAssembler] Queued segment {len(self._segments)}: "
              f"{os.path.basename(image_path)} ({duration:.1f}s)")

    def assemble(self) -> str:
        """
        Builds all queued segments into a single video and writes the .mp4 file.

        Returns:
            Path to the final output .mp4 file.
        """
        if not self._segments:
            raise RuntimeError("No segments added. Call add_segment() before assemble().")

        print(f"\n  [VideoAssembler] Building {len(self._segments)} clips ...")
        clips = self._build_clips()

        print(f"  [VideoAssembler] Concatenating clips with "
              f"{self.transition_duration}s crossfade ...")
        final_video = concatenate_videoclips(clips, method="compose")

        print(f"  [VideoAssembler] Writing video to: {self.output_path}")
        print("  [VideoAssembler] This may take a few minutes ...")
        final_video.write_videofile(
            self.output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,  # Suppress verbose moviepy logs
        )

        # Clean up clip objects from memory
        for clip in clips:
            clip.close()
        final_video.close()

        print(f"  [VideoAssembler] ✓ Video saved: {self.output_path}")
        return self.output_path

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_clips(self) -> list:
        """
        Converts each segment dict into a MoviePy clip (image + audio + crossfade).
        """
        clips = []

        for i, segment in enumerate(self._segments):
            # Load audio and measure its real duration
            audio_clip = AudioFileClip(segment["audio"])

            # Create an image clip that lasts exactly as long as the audio
            image_clip = (
                ImageClip(segment["image"])
                .set_duration(audio_clip.duration)
                .set_audio(audio_clip)
                .set_fps(self.fps)
            )

            # Add a crossfade-in for every clip except the very first one
            if i > 0:
                image_clip = image_clip.crossfadein(self.transition_duration)

            clips.append(image_clip)

        return clips
