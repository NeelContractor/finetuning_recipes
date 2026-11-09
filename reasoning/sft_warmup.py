import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from checkpointing import (
    add_checkpoint_args,
    hub_repo_from_args,
    make_hub_checkpoint_callback,
    resolve_resume_checkpoint,
    upload_final_model,
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    to_sharegpt,
    train_on_responses_only,
)

if sys.platform != "darwin":
    import transformers.utils.generic

    transformers.utils.generic._is_mlx_available = False

from datasets import load_dataset
from datasets.combine import concatenate_datasets
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


SEED = 3407
# Avoid oversubscribing CPUs when each DDP rank spawns its own workers.
WORKERS = 2 if int(os.environ.get("WORLD_SIZE") or "1") > 1 else 8

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
Generate your reasoning first inside <think> and </think> tags. After </think>, generate only the requested final response.
When a structured format such as JSON is requested, the content after </think> must contain only that format, without Markdown fences or additional commentary.
    """


def format_reasoning_response(reasoning: str, output: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n{output.strip()}"


parser = argparse.ArgumentParser(
    description="ChatML reasoning warmup with Unsloth LoRA."
)
parser.add_argument(
    "--base_model_id",
    "-i",
    type=str,
    default="paperbd/smollm_135M_arxiv_cpt",
    help="Path to a model in models/ directory or a HF model ID.",
)
parser.add_argument(
    "--output_model_id",
    "-o",
    type=str,
    default="reasoning_warmup",
    help="Output subdirectory under models/.",
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="paperbd/paper_instructions_300K-v1",
    help="HF dataset containing instruction, input, reasoning, and output.",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    default="reasoning",
    help="HF dataset configuration containing reasoning rows.",
)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--batch_size", "-bs", type=int, default=32)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--epochs", "-e", type=int, default=3)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--conversation_extension", type=int, default=1)
parser.add_argument("--variations", type=int, default=1)
parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
add_checkpoint_args(parser)

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK") or "0"))

output_dir = f"models/{args.output_model_id}"
resume_checkpoint = resolve_resume_checkpoint(
    output_dir,
    explicit=args.resume_from_checkpoint,
    auto=args.resume,
)
hub_repo = hub_repo_from_args(args)

if args.conversation_extension == 1:
    args.variations = 1

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model_id,
    max_seq_length=args.max_seq_length,
    load_in_4bit=args.load_in_4bit,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

dataset = load_dataset(
    args.dataset,
    args.dataset_config,
    split="train",
)
dataset = dataset.map(
    lambda example: {
        "reasoning_response": format_reasoning_response(
            example["reasoning"],
            example["output"],
        )
    }
)
dataset_variations = []

for i in range(args.variations):
    dataset_variations.append(
        to_sharegpt(
            dataset,
            merged_prompt="{instruction}\n\n{input}",
            output_column_name="reasoning_response",
            conversation_extension=args.conversation_extension,
            random_state=SEED + i,
        )
    )
dataset = concatenate_datasets(dataset_variations)
del dataset_variations

dataset = standardize_data_formats(dataset)


def formatting_func(examples, tokenizer):
    # The reasoning and final response remain in one assistant turn. The
    # response-only trainer therefore masks system/user tokens as before.
    convos = examples["conversations"]
    system_part = [{"role": "system", "content": SYSTEM_PROMPT}]
    texts = [
        tokenizer.apply_chat_template(
            system_part + conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in convos
    ]
    return {"text": texts}


dataset = dataset.map(
    lambda examples: formatting_func(examples, tokenizer),
    batched=True,
    remove_columns=dataset.column_names,
)


def fits_context_window(example):
    return (
        len(tokenizer.encode(example["text"], add_special_tokens=False))
        <= args.max_seq_length
    )


rows_before_length_filter = len(dataset)
dataset = dataset.filter(fits_context_window)
filtered_rows = rows_before_length_filter - len(dataset)
print(
    f"Filtered {filtered_rows} rows exceeding "
    f"max_seq_length={args.max_seq_length}"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=args.lora_r,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=args.lora_r >= 64,
    loftq_config=None,
)

if len(dataset) < 2:
    raise ValueError(
        f"Only {len(dataset)} reasoning example(s) fit within "
        f"max_seq_length={args.max_seq_length}; raise --max_seq_length or add data."
    )
split = dataset.train_test_split(
    test_size=1 if len(dataset) < 50 else 0.02, seed=SEED
)
train_dataset = split["train"]
val_dataset = split["test"]

max_grad_norm = 1.0

# torch>=2.10's is_bf16_supported() defaults to including_emulation=True, which
# reports True on T4/P100 and makes the Trainer silently pick a much slower
# emulated-bf16 path. Gate on native support (Ampere / sm80) and use fp16 on
# older GPUs instead.
bf16_supported = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        dataloader_num_workers=WORKERS,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        report_to="none",
        max_grad_norm=max_grad_norm,
        seed=SEED,
        max_length=args.max_seq_length,
        packing=True,
        dataset_num_proc=WORKERS,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
    )
)
if hub_repo:
    trainer.add_callback(
        make_hub_checkpoint_callback(
            repo_id=hub_repo,
            path_prefix=args.output_model_id,
            private=args.hub_private,
            upload_every=args.hub_upload_every_saves,
        )
    )

trainer.train(resume_from_checkpoint=resume_checkpoint)

final_dir = f"{output_dir}/final"
if trainer.is_world_process_zero():
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")
    if hub_repo:
        upload_final_model(
            output_dir,
            hub_repo,
            args.output_model_id,
            private=args.hub_private,
        )
