"""
blip_captioner.py
-----------------
Simple, modular image captioning using BLIP-1 or BLIP-2 from HuggingFace.

Install dependencies:
    pip install torch torchvision transformers pillow requests

Usage:
    from blip_captioner import BLIPCaptioner

    captioner = BLIPCaptioner(model="blip1")          # or model="blip2"
    caption = captioner.caption("path/to/image.jpg")
    print(caption)
"""

from pathlib import Path
from PIL import Image
import requests
import torch
from typing import Optional, Union

# ──────────────────────────────────────────────
# Supported model configs
# ──────────────────────────────────────────────
MODELS = {
    "blip1": {
        "model_id": "Salesforce/blip-image-captioning-large",
        "type": "blip1",
    },
    "blip1-base": {
        "model_id": "Salesforce/blip-image-captioning-base",
        "type": "blip1",
    },
    "blip2": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "type": "blip2",
    },
    "blip2-flan": {
        "model_id": "Salesforce/blip2-flan-t5-xl",
        "type": "blip2",
    },
}


class BLIPCaptioner:
    """
    Wraps BLIP-1 and BLIP-2 for image captioning.

    Args:
        model (str): One of "blip1", "blip1-base", "blip2", "blip2-flan".
                     Default: "blip1"
        device (str): "cuda", "cpu", or None (auto-detect).
        dtype (torch.dtype): Precision. Default float16 on GPU, float32 on CPU.
    """

    def __init__(self, model: str = "blip1", device: Optional[str] = None, dtype:Optional[Union[str,torch.dtype]]=None):
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODELS)}")

        config = MODELS[model]
        self.model_id = config["model_id"]
        self.model_type = config["type"]

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        print(f"[BLIPCaptioner] Loading {model} ({self.model_id}) on {self.device}...")
        self._load_model()
        print("[BLIPCaptioner] Ready.")

    def _load_model(self):
        if self.model_type == "blip1":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            ).to(self.device)

        elif self.model_type == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            ).to(self.device)

        self.model.eval()

    def _load_image(self, source: str | Path) -> Image.Image:
        """Load image from a file path or URL."""
        source = str(source)
        if source.startswith("http://") or source.startswith("https://"):
            image = Image.open(requests.get(source, stream=True).raw).convert("RGB")
        else:
            image = Image.open(source).convert("RGB")
        return image

    def caption(
        self,
        image_source: str | Path,
        prompt: Optional[str] = None,
        max_new_tokens: int = 100,
        num_beams: int = 5,
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image_source: File path or URL to the image.
            prompt: Optional text prompt to condition the caption (e.g. "a photo of").
                    BLIP-2 supports full question-answering prompts too.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width (higher = better but slower).

        Returns:
            Caption string.
        """
        image = self._load_image(image_source)

        inputs = self.processor(
            images=image,
            text=prompt,       # None is fine — unconditional captioning
            return_tensors="pt"
        ).to(self.device, self.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def caption_batch(
        self,
        image_sources: list,
        prompt: str = None,
        max_new_tokens: int = 100,
        num_beams: int = 5,
    ) -> list[str]:
        """
        Caption multiple images. Returns a list of caption strings.
        Processes one at a time (safe for low-VRAM setups).
        """
        return [
            self.caption(src, prompt=prompt, max_new_tokens=max_new_tokens, num_beams=num_beams)
            for src in image_sources
        ]


# ──────────────────────────────────────────────
# Quick demo — run this file directly to test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    TEST_URL = "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=640"

    print("=== BLIP-1 Demo ===")
    cap1 = BLIPCaptioner(model="blip1")

    # Unconditional caption
    result = cap1.caption(TEST_URL)
    print(f"Unconditional: {result}")

    # Conditional (prompted) caption
    result_cond = cap1.caption(TEST_URL, prompt="a photo of")
    print(f"Prompted:      {result_cond}")

    print("\n=== BLIP-1 VQA-style ===")
    # BLIP-1 can also answer questions when given a prompt
    result_qa = cap1.caption(TEST_URL, prompt="Question: What animal is in this photo? Answer:")
    print(f"Q&A:           {result_qa}")

    # Uncomment to test BLIP-2 (needs more VRAM / time):
    # print("\n=== BLIP-2 Demo ===")
    # cap2 = BLIPCaptioner(model="blip2")
    # result2 = cap2.caption(TEST_URL)
    # print(f"BLIP-2: {result2}")
    # result2_qa = cap2.caption(TEST_URL, prompt="Question: What is the dog doing? Answer:")
    # print(f"BLIP-2 Q&A: {result2_qa}")