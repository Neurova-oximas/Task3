"""
imagecaption.py
---------------
Required module as per task specification.
Thin wrapper around BLIPCaptioner — keeps the rest of the codebase
decoupled from the BLIP implementation details.

Usage:
    from imagecaption import caption_image
    text = caption_image(pil_image)
"""

from PIL import Image
from blip_captioner import BLIPCaptioner

# ── Load once at import time (expensive — model weights loaded here) ───────────
# @st.cache_resource equivalent for plain Python: module-level singleton.
# The first import loads the model; subsequent imports reuse it.

#captioner = BLIPCaptioner(model="blip1")

# def caption_image(
#     image: Image.Image,
#     prompt: str | None = None,
#     max_new_tokens: int = 100,
#     num_beams: int = 5,
# ) -> str:
#     """
#     Generate a text caption for a PIL image using BLIP-1.

#     Parameters
#     ----------
#     image          : PIL.Image.Image — RGB image (already converted by app.py)
#     prompt         : optional conditioning prompt, e.g. "a photo of"
#     max_new_tokens : max tokens to generate (passed through to BLIP)
#     num_beams      : beam search width (higher = better quality, slower)

#     Returns
#     -------
#     str — caption string describing the image content
#     """
#     return _captioner.caption(
#         image_source=image,
#         prompt=prompt,
#         max_new_tokens=max_new_tokens,
#         num_beams=num_beams,
#     )

def caption_image(
    image: Image.Image,
    prompt: str | None = None,
    max_new_tokens: int = 100,
    num_beams: int = 5,
) -> str:
    """
    MOCK: returns a fake caption regardless of input.
    Replace the return value with real BLIPCaptioner when ready.
    its commented above just uncomment it and comment this one
    """
    return "a person standing in front of a building holding a sign"