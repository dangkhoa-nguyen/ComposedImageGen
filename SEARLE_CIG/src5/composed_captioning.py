"""
Generate composed captions for CIRCO queries.

Supported models:
- blip
- Qwen/Qwen2.5-VL-3B-Instruct-AWQ
- Qwen/Qwen3-VL-4B-Instruct-FP8

Examples:
    python composed_captioning.py \
        --dataset circo \
        --dataset-path /path/to/CIRCO \
        --model blip \
        --file-name circo_blip_composed.json

    python composed_captioning.py \
        --dataset circo \
        --dataset-path /path/to/CIRCO \
        --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
        --prompt ./prompt.py \
        --file-name circo_qwen_composed.json
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipProcessor

from data_utils import PROJECT_ROOT
from datasets import CIRCODataset
from utils_feat import device
from qwen_vl_utils import process_vision_info


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
QWEN_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "Qwen/Qwen3-VL-4B-Instruct-FP8",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test_submissions" / "circo"


def load_system_prompt(prompt_path: Optional[str]) -> str:
    """
    Load `system_prompt` from a supplied Python file, or use prompt.py
    beside this script when --prompt is omitted.
    """
    if prompt_path is None:
        prompt_file = Path(__file__).resolve().with_name("prompt.py")
    else:
        prompt_file = Path(prompt_path).expanduser().resolve()

    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    module_name = f"_custom_prompt_{abs(hash(str(prompt_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, prompt_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt Python file: {prompt_file}")

    prompt_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(prompt_module)
    except Exception as exc:
        raise RuntimeError(
            f"An error occurred while importing prompt file '{prompt_file}': {exc}"
        ) from exc

    if not hasattr(prompt_module, "system_prompt"):
        raise ValueError(
            f"Prompt file '{prompt_file}' does not define a 'system_prompt' variable."
        )

    system_prompt = getattr(prompt_module, "system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            f"'system_prompt' in '{prompt_file}' must be a non-empty string."
        )

    print(f"Using system prompt from: {prompt_file}")
    return system_prompt.strip()


def get_item_value(item: Dict[str, Any], *keys: str, default=None):
    """Return the first present, non-None field from a dataset item."""
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def to_jsonable(value):
    """Convert common NumPy/PyTorch scalar/container values to JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    model = model.to(device).eval()
    return processor, model


# def load_qwen(model_name: str):
#     """
#     Load a supported Qwen-VL model with the generic Transformers auto classes.
#     device_map='auto' lets Transformers place model layers on available devices.
#     """
#     try:
#         from transformers import AutoModelForImageTextToText

#         model_class = AutoModelForImageTextToText
#     except ImportError:
#         # Compatibility fallback for Transformers versions that predate
#         # AutoModelForImageTextToText.
#         from transformers import AutoModelForVision2Seq

#         model_class = AutoModelForVision2Seq

#     processor = AutoProcessor.from_pretrained(model_name)
#     model = model_class.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto",
#     )
#     model.eval()
#     return processor, model

def load_qwen(model_name: str):
    """Load Qwen2.5-VL or Qwen3-VL using Transformers."""

    from transformers import AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_name)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()
    return processor, model


def generate_blip_caption(processor, model, image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=50)

    caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return caption


# def generate_qwen_composed_caption(
#     processor,
#     model,
#     image: Image.Image,
#     relative_caption: str,
#     system_prompt: str,
# ) -> str:
#     """
#     Ask the vision-language model to describe the imagined target image.
#     Qwen processors build the multimodal input, including the image.
#     """
#     user_text = (
#         "MODIFICATION:\n"
#         f"{relative_caption.strip()}\n\n"
#         "Generate the target-image description according to the system instructions."
#     )

#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": user_text},
#             ],
#         },
#     ]

#     # Qwen-VL processors support the chat template and multimodal content
#     # representation. Use the template's tokenized output where available.
#     try:
#         inputs = processor.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt",
#         )
#     except (TypeError, ValueError):
#         # Compatibility path for processor versions that require a rendered
#         # chat template followed by a regular processor call.
#         prompt_text = processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         inputs = processor(
#             text=[prompt_text],
#             images=[image],
#             return_tensors="pt",
#             padding=True,
#         )

#     # A model loaded with device_map='auto' may be spread across devices.
#     # For standard Qwen-VL deployments, the embedding layer's device is the
#     # appropriate input device.
#     try:
#         input_device = model.get_input_embeddings().weight.device
#     except (AttributeError, RuntimeError):
#         input_device = device

#     inputs = {
#         key: value.to(input_device) if isinstance(value, torch.Tensor) else value
#         for key, value in inputs.items()
#     }

#     with torch.inference_mode():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             do_sample=False,
#         )

#     # Remove prompt tokens so that only newly generated text is decoded.
#     input_ids = inputs.get("input_ids")
#     if input_ids is not None:
#         generated_ids = [
#             output_ids[len(input_id):]
#             for input_id, output_ids in zip(input_ids, generated_ids)
#         ]

#     # Keep special markers temporarily so Qwen3-Thinking's reasoning section
#     # can be discarded before extracting the final answer.
#     output = processor.batch_decode(
#         generated_ids,
#         skip_special_tokens=False,
#         clean_up_tokenization_spaces=True,
#     )[0].strip()
#     if "</think>" in output:
#         output = output.rsplit("</think>", 1)[-1]

#     # Remove any remaining model control tokens and normalize whitespace.
#     output = re.sub(r"<\|[^|]+\|>", " ", output)
#     output = output.replace("<think>", " ").replace("</think>", " ")
#     output = " ".join(output.split()).strip().strip('"').strip("'")
#     if not output:
#         raise RuntimeError(
#             "The Qwen model returned an empty composed caption for a query."
#         )
#     return output


def generate_qwen_composed_caption(
    processor,
    model,
    image: Image.Image,
    relative_caption: str,
    system_prompt: str,
) -> str:
    """Generate a composed target-image caption with Qwen-VL."""

    is_qwen3 = "Qwen3-VL" in getattr(model.config, "_name_or_path", "")

    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {
                    "type": "text",
                    "text": (
                        "MODIFICATION:\n"
                        f"{relative_caption.strip()}\n\n"
                        "Generate the target-image description. "
                        "Return only one concise sentence."
                    ),
                },
            ],
        },
    ]

    # --------------------------------------------------------
    # Render chat template
    # --------------------------------------------------------

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Process image according to model family
    # --------------------------------------------------------

    if is_qwen3:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)

    else:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
        )
        video_metadatas = None

    processor_kwargs = {
        "text": [prompt_text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
        "do_resize": False,
        **video_kwargs,
    }

    if is_qwen3:
        processor_kwargs["video_metadata"] = video_metadatas

    inputs = processor(**processor_kwargs)

    # --------------------------------------------------------
    # Move inputs to the model's input device
    # --------------------------------------------------------

    try:
        input_device = model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        input_device = device

    inputs = {
        key: value.to(input_device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Remove prompt tokens
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_length:]

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # --------------------------------------------------------
    # Remove reasoning section if present
    # --------------------------------------------------------

    if "</think>" in output:
        output = output.rsplit("</think>", 1)[-1]
    elif "<think>" in output:
        output = output.split("<think>", 1)[0]

    # Remove remaining control tokens
    output = re.sub(r"<\|[^|]+\|>", " ", output)
    output = output.replace("<think>", " ").replace("</think>", " ")
    output = " ".join(output.split()).strip().strip('"').strip("'")

    if not output:
        raise RuntimeError(
            "The Qwen model returned an empty composed caption."
        )

    return output



def main():
    parser = argparse.ArgumentParser(description="Generate composed captions for the CIRCO dataset.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--file-name", type=str, default="composed_captions.json",
                        help="Output JSON filename.")
    parser.add_argument("--model", type=str, required=True, choices=["blip", *QWEN_MODELS],
                        help="Captioning model to use.")
    parser.add_argument("--dataset", type=str, required=True, choices=["circo"],
                        help="Dataset to use (currently only circo).")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the CIRCO dataset.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="CIRCO split to process (default: test).")
    parser.add_argument("--prompt", type=str, default=None,
                        help=(
                            "Optional path to a Python file defining `system_prompt`. "
                            "If omitted, uses prompt.py beside this script. Ignored for BLIP."
                        ))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = Path(args.file_name).expanduser()
    if output_file.is_absolute():
        output_path = output_file
    else:
        output_path = output_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_blip = args.model == "blip"

    # BLIP does not use a system prompt. Validate/load a prompt only for Qwen.
    system_prompt = None if is_blip else load_system_prompt(args.prompt)

    # The dataset's transform is not needed for caption generation. An identity
    # transform avoids loading a CLIP model solely to obtain preprocessing.
    dataset = CIRCODataset(
        args.dataset_path,
        args.split,
        "relative",
        preprocess=lambda image: image,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"CIRCO split '{args.split}' contains no queries: {args.dataset_path}"
        )

    if is_blip:
        processor, model = load_blip()
    else:
        processor, model = load_qwen(args.model)

    results = []
    for index in tqdm(range(len(dataset)), desc=f"Generating {args.model} captions"):
        item = dataset[index]

        relative_caption = str(
            get_item_value(item, "relative_caption", default="")
        ).strip()
        if not relative_caption:
            raise ValueError(
                f"Query at dataset index {index} has an empty relative_caption."
            )

        reference_image_path = get_item_value(
            item, "reference_image_path", "reference_path"
        )
        if reference_image_path is None:
            raise KeyError(
                "CIRCODataset items must include 'reference_image_path'. "
                "Ensure the dataset implementation returns this field."
            )

        with Image.open(reference_image_path) as image_file:
            reference_image = image_file.convert("RGB")

            if is_blip:
                image_caption = generate_blip_caption(
                    processor, model, reference_image
                )
                # Preserve the existing BLIP composition logic.
                composed_caption = (
                    f"{image_caption.strip().lower()}, "
                    f"{relative_caption.strip().lower()}"
                )
            else:
                composed_caption = generate_qwen_composed_caption(
                    processor,
                    model,
                    reference_image,
                    relative_caption,
                    system_prompt,
                )

        query_id = get_item_value(item, "query_id", "id", "pair_id")
        if query_id is None:
            raise KeyError(
                "Could not find query ID in CIRCODataset item. "
                "Expected one of: query_id, id, pair_id."
            )

        reference_img_id = get_item_value(
            item, "reference_img_id", "reference_name", "reference_image_id"
        )
        gt_img_ids = get_item_value(
            item, "gt_img_ids", "target_img_ids", "target_image_ids", default=[]
        )

        results.append(
            {
                "id": str(query_id),
                "reference_img_id": to_jsonable(reference_img_id),
                "gt_img_ids": to_jsonable(gt_img_ids),
                "relative_caption": relative_caption,
                "composed_caption": composed_caption,
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} composed captions to: {output_path}")


if __name__ == "__main__":
    main()
