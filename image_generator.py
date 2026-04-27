import os
import re
import gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


class ImageGenerator:
    """
    Generates high-quality storybook images using Stable Diffusion 2.1.
    Optimised for CPU-only machines with 8GB RAM.

    Key optimisations used:
      - SD 2.1 model   : much better anatomy/faces than 1.5
      - DPMSolverMultistep scheduler : high quality in fewer steps (20-25 vs 50)
      - float32        : required for CPU (float16 is GPU only)
      - attention slicing : reduces peak memory usage
      - negative prompt : actively tells SD what to avoid (bad faces, blurry, etc.)
      -768x768        : SD 2.1 native resolution -- better than 512x512
      - gc.collect()   : frees RAM between generations

    Install:
        pip install diffusers transformers accelerate torch pillow
    """

    # SD 2.1 -- significantly better faces and anatomy than 1.5
    DEFAULT_MODEL = "sd2-community/stable-diffusion-2-1"

    # Style suffix for consistent storybook look
    STYLE_SUFFIX = (
        "storybook illustration, children's book art, "
        "detailed digital painting, warm colors, "
        "professional illustration, high quality, "
        "sharp focus, 8k, artstation"
    )

    # Negative prompt -- tells SD exactly what to AVOID
    # This is the biggest improvement for fixing faces/anatomy on CPU
    NEGATIVE_PROMPT = (
        "ugly, deformed, disfigured, bad anatomy, "
        "bad proportions, extra limbs, cloned face, "
        "malformed hands, mutated, poorly drawn face, "
        "poorly drawn hands, missing fingers, extra fingers, "
        "blurry, low quality, low resolution, worst quality, "
        "watermark, signature, text, jpeg artifacts, "
        "out of frame, cropped, gross proportions"
    )

    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "was", "it", "he", "she",
        "they", "that", "this", "as", "by", "from", "had", "have",
        "his", "her", "their", "then", "so", "up", "out", "not",
        "said", "very", "just", "been", "were", "into", "there",
    }

    def __init__(
        self,
        output_dir: str = "output/images",
        model_id: str = DEFAULT_MODEL,
        image_width: int = 768,
        image_height: int = 768,
        num_steps: int = 50,
        guidance_scale: float = 8.5,
    ):
        """
        Args:
            output_dir      : Folder to save generated .png images.
            model_id        : Hugging Face model ID.
            image_width     : Image width in pixels. 768 is SD 2.1 native.
            image_height    : Image height in pixels.
            num_steps       : Denoising steps. 20-25 is the sweet spot for
                              DPMSolverMultistep (quality vs speed on CPU).
            guidance_scale  : How strictly to follow the prompt.
                              7.5-9.0 range -- higher = more prompt-faithful
                              but less creative. 8.5 works well for storybooks.
        """
        self.output_dir     = output_dir
        self.model_id       = model_id
        self.image_width    = image_width
        self.image_height   = image_height
        self.num_steps      = num_steps
        self.guidance_scale = guidance_scale
        self.pipe           = None

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public methods                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """
        Loads SD 2.1 with CPU optimisations.
        First run downloads ~5GB -- cached locally after that.
        """
        print(f"  [ImageGenerator] Loading model '{self.model_id}' ...")
        print("  [ImageGenerator] First run downloads ~5GB -- please wait ...")
        print("  [ImageGenerator] Subsequent runs load from cache (much faster).")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,   # float32 required for CPU
        )

        # DPMSolverMultistep: gets high quality in 20-25 steps
        # (vs 50 steps needed with default DDIM scheduler)
        # This roughly halves generation time on CPU
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to("cpu")

        # Reduces peak RAM usage significantly on CPU
        self.pipe.enable_attention_slicing(slice_size=1)

        print("  [ImageGenerator] Model loaded on CPU with RAM optimisations.")
        print(f"  [ImageGenerator] Resolution: {self.image_width}x{self.image_height}, "
              f"Steps: {self.num_steps}, CFG: {self.guidance_scale}")

    def generate(self, text: str, index: int) -> str:
        """
        Generates a storybook-style image for a text chunk.

        Args:
            text  : Story text chunk to visualise.
            index : Chunk number for file naming.

        Returns:
            Path to the saved .png file.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_path = os.path.join(self.output_dir, f"image_{index:04d}.png")

        if os.path.exists(output_path):
            print(f"  [ImageGenerator] Reusing existing: {output_path}")
            return output_path

        prompt = self._build_prompt(text)
        print(f"  [ImageGenerator] Generating image {index} ...")
        print(f"  [ImageGenerator] Prompt: {prompt[:100]}...")
        print(f"  [ImageGenerator] (CPU generation takes ~5-15 mins per image)")

        # Set a fixed seed for reproducibility -- change to None for random
        generator = torch.Generator().manual_seed(index * 42)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        image: Image.Image = result.images[0]
        image.save(output_path, quality=95)
        print(f"  [ImageGenerator] Saved: {output_path}")

        # Free RAM between generations -- important on 8GB CPU
        gc.collect()

        return output_path

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, text: str) -> str:
        """
        Builds a rich, descriptive prompt from the story text.

        Strategy:
          1. Extract meaningful keywords (filter stop words)
          2. Detect scene elements (character, setting, mood)
          3. Prepend quality boosters and append style suffix
        """
        clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        words = clean.split()

        keywords = [
            w for w in words
            if w not in self._STOP_WORDS and len(w) > 3
        ]

        # Take top 12 keywords -- too many confuses SD
        top_keywords = keywords[:12]

        if not top_keywords:
            top_keywords = [text[:80]]

        # Quality boosters prepended -- SD reads left-to-right
        # so important words should come first
        quality_prefix = "masterpiece, best quality, highly detailed"

        prompt = (
            quality_prefix + ", "
            + ", ".join(top_keywords) + ", "
            + self.STYLE_SUFFIX
        )

        return prompt
