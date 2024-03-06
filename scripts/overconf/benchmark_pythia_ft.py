from functools import partial
from guesslet import PromptDict, CommonSenseQAPrompt, CommonSenseQAQuestionPrompt, FewShotPreppender
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import PeftModel
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import typer


def falcon_tokenize(sample: PromptDict, tokenizer: PreTrainedTokenizer):
    return tokenizer(sample["prompt"], return_length=True)


def answer_key_to_index(sample):
    return {'answer': ord(sample['answer']) - ord('A')}


def main(output_path: str, peft_model_id: str):
    RANDOM_SEED = 38
    model_name = "EleutherAI/pythia-6.9b"
    dataset = load_dataset("tau/commonsense_qa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question_prompter = CommonSenseQAQuestionPrompt()


    # Prepare inference Dataset
    inference_dataset = (
        dataset["validation"]
        # Create prompt string from structured data
        .map(question_prompter, desc="Prompt Creation: ")
        # Prepend few shot strings
        .map(answer_key_to_index, desc='Ans to Idx: ')
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
    model = PeftModel.from_pretrained(model, peft_model_id)

    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    inference_dataloader = DataLoader(
        inference_dataset.map(None, remove_columns=['prompt']), 
        batch_size=16, 
        collate_fn=collator, 
        shuffle=False
    )

    token_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E"])
    token_id_to_assigned_id = {
        token_id: assigned_id
        for assigned_id, token_id in enumerate(token_ids)
    }

    pred_labels = []
    truth_labels = []
    nlls = []
    for batch in tqdm(inference_dataloader):

        with torch.no_grad():
            outputs = model(**{k:v for k, v in batch.items() if k not in set(['length', 'answer'])})
            # -3 is the place where the answer token is located
            logits = outputs.logits[range(outputs.logits.shape[0]), batch['length']-1]
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