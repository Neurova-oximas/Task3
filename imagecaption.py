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


def caption_image(
    image: Image.Image,
    prompt: str | None = None,
    max_new_tokens: int = 100,
    num_beams: int = 5,
) -> str:
    """
    MOCK: returns a fake caption regardless of input.
    Replace with the real implementation below when BLIP is downloaded.
    """
    return "a person standing in front of a building holding a sign"


# ── Real implementation (uncomment when ready) ─────────────────────────────────
#
# from blip_captioner import BLIPCaptioner
#
# _captioner = BLIPCaptioner(model="blip1")
#
# def caption_image(
#     image: Image.Image,
#     prompt: str | None = None,
#     max_new_tokens: int = 100,
#     num_beams: int = 5,
# ) -> str:
#     return _captioner.caption(
#         image_source=image,
#         prompt=prompt,
#         max_new_tokens=max_new_tokens,
#         num_beams=num_beams,
#     )