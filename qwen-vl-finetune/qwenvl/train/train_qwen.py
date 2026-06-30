# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from qwenvl.train.trainer import replace_qwen2_vl_attention_class
except ImportError:
    from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.data.robotwin_processor import (
    QUERY_TOKENS,
    make_robotwin_data_module,
    robotwin_special_tokens,
    save_robotwin_split_manifest,
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.robotwin_model import RobotWinQwenWrapper
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def is_rank0():
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


class SharedTensorSafeTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        unwrapped_model = self.accelerator.unwrap_model(
            self.model, keep_torch_compile=False
        )
        if not isinstance(unwrapped_model, RobotWinQwenWrapper):
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if hasattr(unwrapped_model, "config"):
            unwrapped_model.config.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        os.makedirs(output_dir, exist_ok=True)
        torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        if hasattr(trainer.model, "config"):
            trainer.model.config.save_pretrained(output_dir)


def _get_submodule_if_exists(model, *paths):
    for path in paths:
        current = model
        found = True
        for name in path.split("."):
            if not hasattr(current, name):
                found = False
                break
            current = getattr(current, name)
        if found:
            return current
    return None


def _set_requires_grad(module, requires_grad):
    if module is None:
        return
    for _, p in module.named_parameters():
        p.requires_grad = requires_grad


def set_model(model_args, model):
    visual = _get_submodule_if_exists(model, "visual", "model.visual")
    merger = _get_submodule_if_exists(model, "visual.merger", "model.visual.merger")
    language_model = _get_submodule_if_exists(model, "language_model", "model.language_model")
    lm_head = _get_submodule_if_exists(model, "lm_head")

    _set_requires_grad(visual, model_args.tune_mm_vision)
    _set_requires_grad(merger, model_args.tune_mm_mlp)
    _set_requires_grad(language_model, model_args.tune_mm_llm)
    _set_requires_grad(lm_head, model_args.tune_mm_llm)


def _tokenizer_from_processor(processor):
    return getattr(processor, "tokenizer", processor)


def _add_robotwin_tokens(processor, tokenizer, model):
    special_tokens = robotwin_special_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processor_tokenizer = _tokenizer_from_processor(processor)
    if processor_tokenizer is not tokenizer:
        processor_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer
    model.resize_token_embeddings(len(tokenizer))
    return {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in QUERY_TOKENS.items()
    }


def _enable_query_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            embeddings.weight.requires_grad = True


def _load_robotwin_init_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return
    path = pathlib.Path(checkpoint_path)
    if path.is_dir():
        path = path / "pytorch_model.bin"
    if not path.exists():
        raise FileNotFoundError(f"RobotWin init checkpoint does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    rank0_print(
        "Loaded RobotWin init checkpoint "
        f"from {path} (missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        rank0_print(f"  missing[:20]={missing[:20]}")
    if unexpected:
        rank0_print(f"  unexpected[:20]={unexpected[:20]}")


def _checkpoint_step(path: pathlib.Path):
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _is_valid_resume_checkpoint(path: pathlib.Path) -> bool:
    weight_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    if any((path / name).is_file() for name in weight_files):
        return True
    if any(child.is_dir() and child.name.startswith("global_step") for child in path.iterdir()):
        return True
    return False


def _find_last_valid_checkpoint(output_dir: str):
    checkpoints = sorted(
        pathlib.Path(output_dir).glob("checkpoint-*"),
        key=_checkpoint_step,
        reverse=True,
    )
    for checkpoint in checkpoints:
        if _is_valid_resume_checkpoint(checkpoint):
            return str(checkpoint)
    return None


def train(attn_implementation="flash_attention_2"):
    global local_rank

    if "--save_safetensors" in sys.argv:
        idx = sys.argv.index("--save_safetensors")
        del sys.argv[idx : min(idx + 2, len(sys.argv))]

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.robotwin_data_root:
        if getattr(training_args, "save_safetensors", False):
            rank0_print(
                "Disabling save_safetensors for RobotWin wrapper because Qwen3-VL ties "
                "lm_head and embed_tokens weights."
            )
        training_args.save_safetensors = False

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.model_max_length = training_args.model_max_length

    robotwin_query_token_ids = None
    if data_args.robotwin_data_root:
        if is_rank0():
            split_path = save_robotwin_split_manifest(
                data_args.robotwin_data_root,
                training_args.output_dir,
                test_ratio=data_args.robotwin_test_ratio,
                split_seed=data_args.robotwin_split_seed,
                anno_root=data_args.robotwin_anno_root,
            )
            rank0_print(f"Saved RobotWin split manifest to {split_path}")
        robotwin_query_token_ids = _add_robotwin_tokens(processor, tokenizer, model)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if is_rank0():
            visual = _get_submodule_if_exists(model, "visual", "model.visual")
            core_model = _get_submodule_if_exists(model, "model")
            if visual is not None and hasattr(visual, "print_trainable_parameters"):
                visual.print_trainable_parameters()
            if core_model is not None and hasattr(core_model, "print_trainable_parameters"):
                core_model.print_trainable_parameters()

    if data_args.robotwin_data_root:
        if training_args.robotwin_train_query_embeddings:
            _enable_query_embeddings(model)
        model = RobotWinQwenWrapper(
            model,
            done_loss_weight=training_args.robotwin_done_loss_weight,
            progress_loss_weight=training_args.robotwin_progress_loss_weight,
            replan_loss_weight=training_args.robotwin_replan_loss_weight,
            incident_loss_weight=training_args.robotwin_incident_loss_weight,
        )
        _load_robotwin_init_checkpoint(model, training_args.robotwin_init_checkpoint)
    
    if data_args.robotwin_data_root:
        data_module = make_robotwin_data_module(
            processor,
            data_args=data_args,
            query_token_ids=robotwin_query_token_ids,
        )
    else:
        data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = SharedTensorSafeTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    resume_checkpoint = _find_last_valid_checkpoint(training_args.output_dir)
    if resume_checkpoint:
        logging.info(f"checkpoint found, resume training from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        incomplete_checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if incomplete_checkpoints:
            rank0_print(
                "Ignoring checkpoint directories without model weights: "
                + ", ".join(str(path) for path in incomplete_checkpoints)
            )
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
