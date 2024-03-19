from functools import partial
from guesslet.prompts.cqa_prompt import CommonSenseQAQuestionPrompt, PromptDict, CommonSenseQAPrompt
from guesslet.prompts.few_shot import FewShotPreppender
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import typer


def mistral_tokenize(sample: PromptDict, tokenizer: PreTrainedTokenizer, max_length=None):
    return tokenizer(sample["prompt"])


def answer_key_to_index(sample):
    return {'answer': ord(sample['answer']) - ord('A')}


def main(output_path: str):
    RANDOM_SEED = 38
    model_name = "mistralai/Mistral-7B-v0.1"
    dataset = load_dataset("tau/commonsense_qa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompter = CommonSenseQAPrompt()
    question_prompter = CommonSenseQAQuestionPrompt()


    # Few Shot
    rng = np.random.default_rng(seed=RANDOM_SEED)
    few_shot_indexes = rng.choice(dataset["train"].num_rows, size=3, replace=False)
    fewshot_examples = (
        dataset["train"]
            .shuffle(seed=RANDOM_SEED)
            # Select the quantity of few shot examples
            .select(few_shot_indexes)
            # Create prompt string from structured data
            .map(prompter, desc="Prompt Creation: ")
    )

    few_shot_preppeder = FewShotPreppender(examples=fewshot_examples['prompt'])

    # Prepare inference Dataset
    inference_dataset = (
        dataset["validation"]
        # Create prompt string from structured data
        .map(question_prompter, desc="Prompt Creation: ")
        # Prepend few shot strings
        .map(few_shot_preppeder, desc="Few Shot Prepend: ")
        .map(answer_key_to_index, desc='Ans to Idx: ')
        .map(partial(mistral_tokenize, tokenizer=tokenizer),
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


    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    inference_dataloader = DataLoader(
        inference_dataset.map(None, remove_columns=['prompt']), 
        batch_size=16, 
        collate_fn=collator, 
        shuffle=False
    )

    token_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E"])

    pred_labels = []
    truth_labels = []
    nlls = []
    for batch in tqdm(inference_dataloader):

        with torch.no_grad():
            outputs = model(**{k:v for k, v in batch.items() if k not in set(['length', 'answer'])})
            # -3 is the place where the answer token is located
            logits = outputs.logits[range(outputs.logits.shape[0]), -1]
            # extract logits values at tokens corresponding to letters "a", "b", "c", "d", "e"
            choice_logits = logits[:, token_ids]
            # calculate softmax of the logits and get model label predictions
            probs = F.softmax(choice_logits.float(), dim=1)
            pred_ids = probs.argmax(-1)

            labels = batch["answer"]

            incorrects = pred_ids != labels
            nll = F.cross_entropy(choice_logits.float(), labels, reduction="none")

            pred_labels.append(pred_ids.cpu())
            truth_labels.append(labels.cpu())
            nlls.append(nll[incorrects].cpu())

    pred_labels = np.concatenate(pred_labels)
    truth_labels = np.concatenate(truth_labels)
    nlls = np.concatenate(nlls)
    
    np.savez(output_path, pred_labels=pred_labels, truth_labels=truth_labels, nlls=nlls)


if __name__ == '__main__':
    typer.run(main)