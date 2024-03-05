from functools import partial
from typing import Dict, Iterable, TypedDict
from datasets import Dataset, Features, ClassLabel, Sequence, Value
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from guesslet import CommonSenseQAPrompt
from transformers import (
    DataCollatorForLanguageModeling, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizer, 
    PreTrainedModel,
    BatchEncoding,
    BitsAndBytesConfig
)
from transformers.modeling_outputs import CausalLMOutput
from datasets import load_dataset

from guesslet import PromptDict
from pathlib import Path


class BatchInput(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor


class FeatureDict(TypedDict):
    sentence: str
    logits: torch.FloatTensor


def falcon_tokenize(sample: PromptDict, tokenizer: PreTrainedTokenizer) -> BatchEncoding: 
    return tokenizer(sample["prompt"])


def process_batch(batch: BatchInput, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> FeatureDict:
    with torch.no_grad():
        outputs: CausalLMOutput = model(**{k:v for k, v in batch.items() if k not in set(['length'])})
        
        logits = outputs.logits.clone().cpu()
        
        tokens = batch['input_ids'].clone().cpu()
        sentence = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    return {
        'sentence': sentence,
        'logits': logits
    }


def gen_data(model: PreTrainedModel, dataloader: DataLoader) -> Iterable[FeatureDict]:
    for batch in tqdm(dataloader):
        yield process_batch(batch)


def main():
    save_path = ""
    model_name = "tiiuae/falcon-7b"
    RANDOM_SEED = 38
    # Load Dataset
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
    dataloader = DataLoader(
        dataset['validation'].map(None, remove_columns=['prompt']), 
        batch_size=16, 
        collate_fn=data_collator, 
        shuffle=False
    )

    # Load Model
    # Pretrained Mistral loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )

    dataset = Dataset.from_generator(
        partial(gen_data, model=model, dataloader=dataloader), 
        features=Features({
            "sentence": Value("string"),
            "logits": Sequence(Value("float"))
        })
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)


if __name__ == '__main__':
    main()