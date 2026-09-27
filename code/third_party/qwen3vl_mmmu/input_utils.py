"""Qwen multimodal input preparation, extracted without changing the function.

Source: QwenLM/Qwen3-VL, evaluation/mmmu/run_mmmu.py
Commit: 96588727e44c78b25ba03ea03b8e12f7e64fd0da
License: Apache-2.0 (see LICENSE in this directory).
"""
import os
from qwen_vl_utils import process_vision_info

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for vLLM (following the examples in README.md).
    
    Args:
        messages: List of messages in standard conversation format
        processor: AutoProcessor instance
    
    Returns:
        dict: Input format required by vLLM
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }
