from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local LoRA SFT training.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3-4b-quant-lora"))
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length. Use 0 or a negative value to disable truncation.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def load_yaml_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_args, remaining = parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**load_yaml_config(config_args.config))
    args = parser.parse_args()
    args.train_file = Path(args.train_file)
    args.val_file = Path(args.val_file)
    args.output_dir = Path(args.output_dir)
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_tokenize_fn(tokenizer, max_length: int | None):
    should_truncate = max_length is not None and max_length > 0

    def encode_text(text: str) -> dict:
        kwargs = {
            "add_special_tokens": False,
            "truncation": should_truncate,
        }
        if should_truncate:
            kwargs["max_length"] = max_length
        return tokenizer(text, **kwargs)

    def tokenize(example: dict) -> dict:
        messages = example["messages"]
        prompt_messages = messages[:-1]
        assistant_message = messages[-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + assistant_message["content"]
        if tokenizer.eos_token:
            full_text += tokenizer.eos_token

        prompt_ids = encode_text(prompt_text)["input_ids"]
        encoded = encode_text(full_text)

        input_ids = encoded["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        return encoded

    return tokenize


def main() -> None:
    args = parse_args()

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_normalize_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(load_jsonl(args.train_file))
    val_dataset = Dataset.from_list(load_jsonl(args.val_file))
    tokenize = build_tokenize_fn(tokenizer, args.max_length)
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        save_safetensors=True,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    with (args.output_dir / "log_history.json").open("w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


def _normalize_target_modules(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported target_modules type: {type(value).__name__}")


if __name__ == "__main__":
    main()
