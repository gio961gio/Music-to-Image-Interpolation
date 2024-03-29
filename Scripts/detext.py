import pytesseract
from dataclasses import dataclass
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw
from typing import Sequence

@dataclass
class TextBox:
  # (x, y) is the top left corner of a rectangle; the origin of the coordinate system is the top-left of the image.
  # x denotes the vertical axis, y denotes the horizontal axis (to match the traditional indexing in a matrix).
  x: int
  y: int
  h: int
  w: int
  text: str = None


class TextDetector:
  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    pass


class TesseractTextDetector(TextDetector):
  """Uses the `tesseract` OCR library from Google to do text detection."""

  def __init__(self, tesseract_path: str):
    """
    Args:
      tesseract_path: The path where the `tesseract` library is installed, e.g. "/usr/bin/tesseract".
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    image = Image.open(image_filename)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxes = [TextBox(l, top, w, h, text)
             for l, top, w, h, text in zip(data["left"], data["top"], data["width"], data["height"], data["text"])
             if text.strip()]
    return boxes


def overlap_x(box1: TextBox, box2: TextBox) -> int:
    return min(box1.x + box1.h, box2.x + box2.h) - max(box1.x, box2.x)


def overlap_y(box1: TextBox, box2: TextBox) -> int:
    return min(box1.y + box1.w, box2.y + box2.w) - max(box1.y, box2.y)


def boxes_intersect(box1: TextBox, box2: TextBox) -> bool:
    return overlap_x(box1, box2) > 0 and overlap_y(box1, box2) > 0


class Inpainter:
  """Interface for in-painting models."""
  # TODO(julia): Run some experiments to determine the best prompt.
  # DEFAULT_PROMPT = "plain background"
  DEFAULT_PROMPT = "The fire mesmerizes with its captivating dance of vibrant colors, rising with a wild grace towards the sky."

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    pass


class StableDiffusionInpainter(Inpainter):
  """Abstract class for Stable Diffusion inpainters; suppoerts any input image size. Children must implement `call_model`."""

  def call_model(self, prompt: str, image: Image, mask: Image) -> Image:
    pass  # To be implemented by children.

  def _tile_has_text_box(self, crop_x: int, crop_y: int, crop_size: int, text_boxes: Sequence[TextBox]):
    # Turn the tile into a TextBox just so that we can reuse utils.boxes_intersect
    crop_box = TextBox(crop_x, crop_y, crop_size, crop_size)
    return any([boxes_intersect(crop_box, text_box) for text_box in text_boxes])

  def _pad_to_size(self, image, size):
    new_image = Image.new(image.mode, (size, size), color=(0, 0, 0))
    new_image.paste(image)
    return new_image

  def _make_mask(self, text_boxes: Sequence[TextBox], height: int, width: int, mode: str) -> Image:
    """Returns a black image with white rectangles where the text boxes are."""
    num_channels = len(mode)
    background_color = tuple([0] * num_channels)
    mask_color = tuple([255] * num_channels)

    mask = Image.new(mode, (width, height), background_color)
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=mask_color)
    return mask

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    image = Image.open(in_image_path)
    mask_image = self._make_mask(text_boxes, image.height, image.width, image.mode)

    # SD only accepts images that are exactly 512 x 512.
    SD_SIZE = 512

    if image.height == SD_SIZE and image.width == SD_SIZE:
      out_image = self.call_model(prompt=prompt, image=image, mask=mask_image)
    else:
      # Break the image into 512 x 512 tiles. In-paint the tiles that contain text boxes.
      out_image = image.copy()

      # Used for the final out_image.paste; required to be in mode L.
      mask_binary = self._make_mask(text_boxes, image.height, image.width, "L")

      for x in range(0, image.height, SD_SIZE):
        for y in range(0, image.width, SD_SIZE):
          if self._tile_has_text_box(x, y, SD_SIZE, text_boxes):
            crop_x1 = min(x + SD_SIZE, image.height)
            crop_y1 = min(y + SD_SIZE, image.width)
            crop_box = (x, y, crop_x1, crop_y1)

            in_tile = self._pad_to_size(image.crop(crop_box), SD_SIZE)
            in_mask = self._pad_to_size(mask_image.crop(crop_box), SD_SIZE)
            out_tile = self.call_model(prompt=prompt, image=in_tile, mask=in_mask)
            out_tile = out_tile.crop((0, 0, crop_x1 - x, crop_y1 - y))
            out_mask = mask_binary.crop(crop_box)
            out_image.paste(out_tile, (x, y), out_mask)

    out_image.save(out_image_path)


class LocalSDInpainter(StableDiffusionInpainter):
  """Uses a local Stable Diffusion model from HuggingFace for in-painting."""

  def __init__(self, pipe: StableDiffusionInpaintPipeline = None):
    if pipe is None:
      if not torch.cuda.is_available():
        raise Exception("You need a GPU + CUDA to run this model locally.")

      self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
          "stabilityai/stable-diffusion-2-inpainting",
          revision="fp16",
          torch_dtype=torch.float16).to("cuda")
    else:
      self.pipe = pipe

  def call_model(self, prompt: str, image: Image, mask: Image) -> Image:
    return self.pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=50).images[0]


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter):
        self.text_detector = text_detector
        self.inpainter = inpainter

    def detextify(self, in_image_path: str, out_image_path: str, prompt=Inpainter.DEFAULT_PROMPT, max_retries=5):
        to_inpaint_path = in_image_path
        for i in range(max_retries):
            print(f"Iteration {i} of {max_retries} for image {in_image_path}:")

            print(f"\tCalling text detector...")
            text_boxes = self.text_detector.detect_text(to_inpaint_path)
            print(f"\tDetected {len(text_boxes)} text boxes.")

            if not text_boxes:
                break

            print(f"\tCalling in-painting model...")
            self.inpainter.inpaint(to_inpaint_path, text_boxes, prompt, out_image_path)
            import os
            assert os.path.exists(out_image_path)
            to_inpaint_path = out_image_path

