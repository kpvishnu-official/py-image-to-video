import os
import re
import gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


class ImageGenerator:
    """
    Optimized Image Generator for low-end CPU (8GB RAM)
    - Faster inference
    - Better prompt quality
    - Improved character consistency
    """

    # 🔥 Best lightweight + high-quality model
    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

    STYLE = (
        "anime illustration, cinematic lighting, soft shadows, "
        "highly detailed face, expressive eyes, clean line art, "
        "consistent character design, vibrant colors, studio quality"
    )

    NEGATIVE_PROMPT = (
        "ugly, deformed, bad anatomy, extra limbs, extra fingers, "
        "missing fingers, poorly drawn face, blurry, low quality, "
        "jpeg artifacts, watermark, text, logo, 3d render, realistic photo"
    )

    _STOP_WORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","is","was","it","he","she","they","that","this","as","by",
        "from","had","have","his","her","their","then","so","up","out",
        "not","said","very","just","been","were","into","there","could",
        "would","should","when","what","which","who","will","about",
    }

    _SCENE_WORDS = {
        "forest","castle","garden","river","mountain","cave","village",
        "house","tree","flower","field","path","bridge","lake","ocean",
        "sky","door","room","library","market","night","morning","sunset",
        "rain","snow","fire","light","dark","magic","golden","ancient",
        "mysterious","enchanted","running","walking","sitting","standing",
        "looking","holding","flying","jumping","crying","laughing",
        "smiling","afraid","brave","happy","lost","found","searching",
    }

    def __init__(
        self,
        output_dir: str = "output/images",
        image_width: int = 512,
        image_height: int = 512,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
    ):
        self.output_dir     = output_dir
        self.image_width    = image_width
        self.image_height   = image_height
        self.num_steps      = num_steps
        self.guidance_scale = guidance_scale
        self.pipe           = None

        os.makedirs(self.output_dir, exist_ok=True)

    # 🔥 Load model (CPU optimized)
    def load_model(self):
        print("[ImageGenerator] Loading optimized model...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.DEFAULT_MODEL,
            torch_dtype=torch.float32,
        )

        # Better scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to("cpu")

        # 🔧 Memory optimizations
        self.pipe.enable_attention_slicing()

        print("[ImageGenerator] Model ready.")

    # 🎯 Main generate function
    def generate(self, text: str, index: int, character_anchor: str = "") -> str:
        if self.pipe is None:
            raise RuntimeError("Call load_model() first.")

        output_path = os.path.join(self.output_dir, f"image_{index:04d}.png")

        if os.path.exists(output_path):
            print(f"[Reuse] {output_path}")
            return output_path

        prompt = self._build_prompt(text, character_anchor)

        print(f"\n[Generating Image {index}]")
        print(f"Prompt: {prompt[:120]}...")

        generator = torch.Generator().manual_seed(42 + index)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        result.images[0].save(output_path, quality=95)

        print(f"[Saved] {output_path}")

        gc.collect()
        return output_path

    # 🔁 Regenerate with new seed
    def regenerate(self, text: str, index: int, character_anchor: str = "", seed: int = 123):
        output_path = os.path.join(self.output_dir, f"image_{index:04d}.png")

        if os.path.exists(output_path):
            os.remove(output_path)

        prompt = self._build_prompt(text, character_anchor)

        generator = torch.Generator().manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        result.images[0].save(output_path, quality=95)

        gc.collect()
        return output_path

    # 🧠 Smart prompt builder
    def _build_prompt(self, text: str, character_anchor: str) -> str:
        clean  = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        words  = clean.split()

        scene = [w for w in words if w in self._SCENE_WORDS][:6]
        others = [
            w for w in words
            if w not in self._STOP_WORDS
            and w not in self._SCENE_WORDS
            and len(w) > 3
        ][:4]

        keywords = ", ".join(scene + others) or text[:60]

        anchor_part = f"{character_anchor}, " if character_anchor else ""

        # 🔥 Key improvements here
        return (
            "masterpiece, best quality, ultra detailed, "
            "same character, same face, consistent appearance, "
            "1girl, "
            + anchor_part
            + keywords + ", "
            + self.STYLE
        )
