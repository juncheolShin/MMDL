"""Run one joint visual/attention-LoRA training arm and export its public-dev checkpoint."""
import argparse
import math
import os
import sys
import tomllib
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.utils import check_cuda_p2p_ib_support
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments, set_seed

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import RECIPE, check_data, file_sha256, main_rank, read_json, require, verified_source, write_json
from training.data import MultimodalCollator, RetentionCollator, RetentionDataset, VisionConversationDataset, verify_prepared_images
from training.model import ALPHA, RANK, configure_attention, standalone
from training.teacher import PROTOCOL
from training.trainer import EvaluateFinalStep, RetentionTrainer


def verify_inputs(args):
    check_data(args.data_dir)
    source = verified_source(args.model)
    must_match = {
        'model_sha256': source, 'config_sha256': file_sha256(args.config),
        'data_manifest_sha256': file_sha256(args.data_dir / 'data_manifest.json'),
        'input_sha256': {s: file_sha256(args.data_dir / f'{s}.jsonl') for s in ('train', 'dev')},
    }
    teacher = read_json(args.teacher_dir / 'manifest.json')
    require(teacher['protocol'] == PROTOCOL and teacher['temperature'] == 1 and teacher['bindings'] == must_match,
            'teacher was scored with different data, model, preprocessing, or temperature')
    for split in ('train', 'dev'):
        require(file_sha256(args.teacher_dir / f'{split}.jsonl') == teacher['outputs'][split]['sha256'],
                f'{split} teacher cache changed after scoring')
    verify_prepared_images([args.data_dir / f'{s}.jsonl' for s in ('train', 'dev')], args.data_dir)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            f'use an empty output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return source, teacher['outputs']


def train(args):
    source, teacher = main_rank(lambda: verify_inputs(args))
    with args.config.open('rb') as stream:
        recipe = tomllib.load(stream)
    settings = recipe['training']
    lora = recipe['lora']
    require((lora['rank'], lora['alpha'], lora['dropout']) == (RANK, ALPHA, 0.0), 'unsupported LoRA recipe')
    set_seed(settings['seed'])
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    batch_size = settings['per_device_train_batch_size']
    collator = RetentionCollator(MultimodalCollator(
        processor.tokenizer.pad_token_id, processor.image_token_id, processor.image_processor.merge_size,
        max_batch_size=batch_size))
    datasets = {}
    for split in ('train', 'dev'):
        data = VisionConversationDataset(args.data_dir / f'{split}.jsonl', processor, args.data_dir, settings)
        labeled = RetentionDataset(data, args.teacher_dir / f'{split}.jsonl')
        require(len(data) == teacher[split]['count'] and
                sum(correct for _, correct, _ in labeled.teacher) == teacher[split]['correct'],
                f'{split} cached teacher decisions disagree with row labels')
        datasets[split] = labeled

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    scope = configure_attention(model)
    require(len(scope['visual_names']) == 315 and len(scope['targets']) == 144 and len(model.model.visual.blocks) == 24,
            'unexpected published Qwen topology')
    model.config.use_cache = model.config.text_config.use_cache = False
    excluded = {'min_pixels', 'max_pixels', 'model_max_length'}
    training = TrainingArguments(
        output_dir=str(args.output_dir), bf16=True, per_device_eval_batch_size=batch_size,
        gradient_checkpointing_kwargs={'use_reentrant': False}, lr_scheduler_type='cosine', optim='adamw_torch',
        eval_strategy='steps', save_strategy='steps', load_best_model_at_end=True,
        metric_for_best_model='eval_selection_score', greater_is_better=True,
        save_only_model=True, save_total_limit=2, save_safetensors=True, logging_first_step=True,
        data_seed=settings['seed'], report_to='none', dataloader_pin_memory=True, remove_unused_columns=False,
        ddp_find_unused_parameters=False, prediction_loss_only=False, ddp_bucket_cap_mb=25,
        **{key: value for key, value in settings.items() if key not in excluded})
    trainer = RetentionTrainer(
        model=model, args=training, train_dataset=datasets['train'], eval_dataset=datasets['dev'],
        data_collator=collator, distill_weight=args.distill_weight, lora_learning_rate=lora['learning_rate'],
        callbacks=[EvaluateFinalStep()])
    optimizer = trainer.create_optimizer()
    require({id(p) for group in optimizer.param_groups for p in group['params']} ==
            {id(p) for p in model.parameters() if p.requires_grad}, 'incorrect optimizer scope')
    if trainer.is_world_process_zero():
        write_json(args.output_dir / 'training_config.json', {
            'recipe': recipe, 'distill_weight': args.distill_weight,
            'world_size': dist.get_world_size(), 'train_examples': len(datasets['train']),
            'dev_examples': len(datasets['dev']), 'teacher': teacher, 'trainable_parameters': scope['trainable_parameters'],
            'visual_tensors': len(scope['visual_names']), 'attention_targets': len(scope['targets']),
            'teacher_manifest_sha256': file_sha256(args.teacher_dir / 'manifest.json'),
            'config_sha256': file_sha256(args.config)})
    trainer.train()
    state = trainer.state
    require(state.best_model_checkpoint and math.isfinite(state.best_metric), 'no valid public-dev checkpoint')
    trainer.save_state()

    def export():
        arguments = args.model, Path(state.best_model_checkpoint), args.output_dir, scope['targets'], source
        written = standalone(*arguments)
        require(written == standalone(*arguments, verify=True), 'export verification failed')
        write_json(args.output_dir / 'export.json', {**written, 'best_global_step': state.best_global_step})
        print(f'Exported checkpoint {state.best_global_step} to {args.output_dir}', flush=True)

    main_rank(export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Published ICoT merged Hugging Face directory')
    parser.add_argument('--data-dir', type=Path, required=True, help='Prepared public train/dev directory')
    parser.add_argument('--teacher-dir', type=Path, required=True, help='Public train/dev cache from teacher.py')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for logs, checkpoints and export')
    parser.add_argument('--distill-weight', type=float, required=True, help='Coefficient of teacher-correct option KL')
    parser.add_argument('--config', type=Path, default=RECIPE, help='Recipe used to produce the teacher cache')
    args = parser.parse_args()
    require(math.isfinite(args.distill_weight) and args.distill_weight >= 0, 'invalid distillation weight')
    require(int(os.environ.get('WORLD_SIZE', 0)) == 4, 'this recipe uses torchrun with four GPUs')
    # Mirror accelerate launch when torchrun is used with RTX hardware lacking P2P/IB.
    if not check_cuda_p2p_ib_support():
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    try:
        train(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
