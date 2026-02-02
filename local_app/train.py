import torch
import gc
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

MAX_SEQ_LENGTH = 2048
DTYPE = None # Auto-detect
LOAD_IN_4BIT = True
BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"

def train_sql_adapter():
    print("\n=== Training SQL Adapter ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    dataset = load_dataset("b-mc2/sql-create-context", split="train[:100]")
    
    def formatting_prompts_func(examples):
        instructions = examples["question"]
        inputs       = examples["context"]
        outputs      = examples["answer"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=30,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            output_dir="outputs_sql",
        ),
    )
    
    trainer.train()
    
    print("Saving SQL Adapter to ./sql_adapter")
    model.save_pretrained("sql_adapter")
    tokenizer.save_pretrained("sql_adapter")

    # Cleanup Memory
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.")

def train_shakespeare_adapter():
    print("\n=== Training Shakespeare Adapter ===")
    # Reload model from scratch for the second adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        random_state=3407,
    )

    dataset = load_dataset("Trelis/tiny-shakespeare", split="train[:100]")

    def formatting_func(examples):
        text = examples["Text"]
        return { "text" : [f"### Style: Shakespeare\n### Content: {t}" for t in text] }

    dataset = dataset.map(formatting_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=30,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            output_dir="outputs_creative",
        ),
    )
    
    trainer.train()
    
    print("Saving Creative Adapter to ./creative_adapter")
    model.save_pretrained("creative_adapter")
    tokenizer.save_pretrained("creative_adapter")
    
    # Cleanup
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_sql_adapter()
    train_shakespeare_adapter()
    print("\nAll training complete! Ready to start vLLM server.")