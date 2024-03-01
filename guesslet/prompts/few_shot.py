from typing import List, TypedDict, Any
from dataclasses import dataclass
from cqa_prompt import PromptDict


@dataclass
class FewShotPreppender:
    examples: List[str]

    def __call__(self, sample: PromptDict) -> PromptDict:
        context = ""

        for example in self.examples:
            context += example + "\n\n\n" # Close the open bracket and add newlines between questions

        return { "prompt": context + sample["prompt"] }