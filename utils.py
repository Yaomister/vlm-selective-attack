"""
Shared between the scripts in this folder.
"""

import re

model_ids = {
    "LLaVA-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "LLaVA-NeXT": "llava-hf/llama3-llava-next-8b-hf",
    "InternVL": "OpenGVLab/InternVL3-8B-hf",
    "Qwen-VL": "Qwen/Qwen2.5-VL-7B-Instruct",
}

sweep_result_file_pattern = re.compile(
    r"results_(?P<pooling>.+?)_(?P<layer_from_last>-?\d+)_(?P<model>.+?)"
    r"_mu(?P<mu>[\d.]+)_epsilon(?P<epsilon>[\d.]+)_(?P<direction>-?[\d.]+)"
    r"_(?P<pair_id>\d+)_(?P<image_type>[a-z]+)_steps(?P<pgd_steps>\d+)\.json"
)


def prepare_inputs(processor, image, prompt, device):
    """Prepare image and prompt as inputs for the VLMs."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # apply the conversation format for the prompt
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    # add the image
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def parse_safety_response(text):
    text = str(text).lower().strip()
    match = re.match(r"\W*(yes|no)\b", text)

    if match:
        return match.group(1)
    else:
        if any(p in text for p in UNSURE):
            return "unclear"
        elif any(p in text for p in YES):
            return "yes"
        elif any(p in text for p in NO):
            return "no"
        else:
            return "unclear"
