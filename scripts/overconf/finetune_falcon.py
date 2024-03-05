from .benchmark_falcon import falcon_tokenize
from guesslet.prompts.cqa_prompt import PromptDict, CommonSenseQAPrompt
from transformers import (
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model
)
import torch
import numpy as np
from datasets import load_dataset
from functools import partial


def main():
    model_name = "tiiuae/falcon-7b"
    RANDOM_SEED = 38
    dataset = load_dataset("tau/commonsense_qa")
    model_name = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompter = CommonSenseQAPrompt()

    # Prepare inference Dataset
    dataset = (
        dataset
        # Create prompt string from structured data
        .map(prompter, desc="Prompt Creation: ")
        .map(partial(falcon_tokenize, tokenizer=tokenizer),
            batched=True,
            remove_columns=[
                'id',
                'question',
                'question_concept',
                'choices',
                'answerKey'
            ],
            desc="Tokenize: "
        )
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        logging_steps=2,
        eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        report_to=["wandb", "tensorboard"],
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={
            "num_cycles": 10
        }
    )

    # Model loading with 4-bit for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,
        target_modules="all-linear"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(10)])
    trainer.train()


if __name__ == '__main__':
    main()