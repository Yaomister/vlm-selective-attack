"""
Shared between the scripts in this folder.
"""

model_ids = {
    "LLaVA-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "LLaVA-NeXT": "llava-hf/llama3-llava-next-8b-hf",
    "InternVL": "OpenGVLab/InternVL3-8B-hf",
    "Qwen-VL": "Qwen/Qwen2.5-VL-7B-Instruct",
}


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
