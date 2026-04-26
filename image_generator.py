import os
import re
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class ImageGenerator:
    """
    Generates images from text using Stable Diffusion (local).
    Uses the Hugging Face `diffusers` library.

    Requirements:
        pip install diffusers transformers accelerate torch pillow

    GPU (NVIDIA) is strongly recommended. On CPU each image can take 10-20 minutes.
    """

    # Default model — a popular, well-tested Stable Diffusion checkpoint.
    # You can swap this for any other model on Hugging Face Hub.
    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

    # Style suffix appended to every prompt to get a consistent visual style.
    STYLE_SUFFIX = (
        "digital art, storybook illustration, "
        "warm lighting, highly detailed, fantasy art"
    )

    # Words that should be stripped before sending to SD (not useful as image prompts)
    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "was", "it", "he", "she",
        "they", "that", "this", "as", "by", "from", "had", "have",
        "his", "her", "their", "then", "so", "up", "out", "not",
    }

    def __init__(
        self,
        output_dir: str = "output/images",
        model_id: str = DEFAULT_MODEL,
        image_width: int = 512,
        image_height: int = 512,
    ):
        """
        Args:
            output_dir    : Folder where generated .png images will be saved.
            model_id      : Hugging Face model ID for Stable Diffusion.
            image_width   : Width of generated images in pixels.
            image_height  : Height of generated images in pixels.
        """
        self.output_dir = output_dir
        self.model_id = model_id
        self.image_width = image_width
        self.image_height = image_height
        self.pipe = None  # Loaded lazily via load_model()
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public methods                                                       #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """
        Downloads (first time) and loads the Stable Diffusion model into memory.
        Automatically uses GPU (CUDA) if available, otherwise falls back to CPU.
        Call this once before calling generate() in a loop.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"  [ImageGenerator] Loading model '{self.model_id}' on {device.upper()} ...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        self.pipe = self.pipe.to(device)

        # Reduce VRAM usage on GPU — safe to call even on CPU (no-op)
        if device == "cuda":
            self.pipe.enable_attention_slicing()

        print(f"  [ImageGenerator] Model loaded on {device.upper()}.")

    def generate(self, text: str, index: int) -> str:
        """
        Generates an image for a given text chunk.

        Args:
            text  : The story text chunk to visualise.
            index : Chunk number (used for naming the file).

        Returns:
            Path to the saved .png image file.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_path = os.path.join(self.output_dir, f"image_{index:04d}.png")

        # Skip regenerating if already exists (saves time on re-runs)
        if os.path.exists(output_path):
            print(f"  [ImageGenerator] Reusing existing image: {output_path}")
            return output_path

        prompt = self._build_prompt(text)
        print(f"  [ImageGenerator] Generating image {index} with prompt:\n    '{prompt}'")

        result = self.pipe(
            prompt=prompt,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=30,   # Higher = better quality but slower
            guidance_scale=7.5,       # How closely to follow the prompt (7-9 is good)
        )
        image: Image.Image = result.images[0]
        image.save(output_path)
        print(f"  [ImageGenerator] Saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, text: str) -> str:
        """
        Extracts the most meaningful keywords from the text and
        formats them into a Stable Diffusion image prompt.

        Strategy:
          1. Remove punctuation.
          2. Filter out common stop words.
          3. Take the top-N most meaningful words.
          4. Append a style suffix for a consistent visual look.
        """
        # Remove non-alphabetic characters
        clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        words = clean.split()

        # Filter stop words and very short words
        keywords = [w for w in words if w not in self._STOP_WORDS and len(w) > 3]

        # Take up to 15 keywords to keep the prompt focused
        top_keywords = keywords[:15]

        if not top_keywords:
            # Fallback: just use the first 80 characters of raw text
            top_keywords = [text[:80]]

        prompt = ", ".join(top_keywords) + ", " + self.STYLE_SUFFIX
        return prompt
