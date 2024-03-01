from dataclasses import dataclass
from functools import partial
from datasets import load_dataset
from typing import List, TypedDict, Any
import torch
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
import numpy as np
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F

################################################################################
### Typings                                                                    #
################################################################################
class MCQChoices(TypedDict):
    label: str
    text: str


class PromptInputData(TypedDict):
    id: int
    question: str
    question_concept: str
    choices: MCQChoices
    answerKey: str


class PromptOutputData(TypedDict):
    id: int
    question: str
    question_concept: str
    choices: MCQChoices
    answerKey: str
    prompt: str


class PromptDict(TypedDict):
    prompt: str
    

class PromptQuestion(TypedDict):
    prompt: str
    answer: str


################################################################################
### Prompting Strategies                                                       #
################################################################################
class CommonSenseQAPrompt:

    def __call__(self, sample: PromptInputData) -> PromptDict:
        """
        Create text prompts of Multiple choice questions

        Paramterers
        -----------
        sample: PromptOutputData
        """
        question = sample['question']
        answer = sample['answerKey'].upper()
        choices = "\n".join([
            f"{label.upper()}. {text}"
            for label, text in zip(
                sample["choices"]["label"],
                sample["choices"]["text"]
            )
        ])

        prompt = (
            f"Question: {question}\n"
            f"{choices}\n"
            f"Answer:{answer}"
        )

        return {'prompt': prompt}


class CommonSenseQAQuestionPrompt:

    def __call__(self, sample: PromptInputData) -> PromptQuestion:
        """
        Create text prompts of Multiple choice questions

        Paramterers
        -----------
        sample: PromptOutputData
        """
        question = sample['question']
        answer = sample['answerKey'].upper()
        choices = "\n".join([
            f"{label.upper()}. {text}"
            for label, text in zip(
                sample["choices"]["label"],
                sample["choices"]["text"]
            )
        ])

        prompt = (
            f"Question: {question}\n"
            f"{choices}\n"
            f"Answer:"
        )

        return {'prompt': prompt, 'answer': answer}
    


class CommonSenseQAPromptParanthesis:

    def __call__(self, sample: PromptInputData) -> dict[str, str]:
        """
        Create text prompts of Multiple choice questions

        Paramterers
        -----------
        sample: PromptOutputData
        """
        # Initialize static strings for the prompt template
        QUESTION_KEY = "Q:"
        CHOICES_KEY = "Answer Choices:"
        ANSWER_KEY = "A:"

        # Combine a prompt with the static strings
        question = f"{QUESTION_KEY} {sample['question']}"
        choices = f"{CHOICES_KEY} \n" + "".join([
                f"({label.lower()}) {text.lower()}\n"
                for label, text in zip(
                    sample["choices"]["label"],
                    sample["choices"]["text"]
                )
        ])
        answer = f"{ANSWER_KEY} ({sample['answerKey'].lower()})"

        prompt = f"{question}\n{choices}\n{answer}"

        return {'prompt': prompt}

