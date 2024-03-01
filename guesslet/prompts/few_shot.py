from typing import List, TypedDict, Any
from dataclasses import dataclass
from .cqa_prompt import PromptDict


@dataclass
class FewShotPreppender:
    """
    A preppender designed for few-shot learning scenarios, where examples are prepended to a given prompt to provide context.

    This class is utilized to create a richer prompt by adding examples before the actual question or task. This technique is commonly used in few-shot learning to improve the performance of models by giving them examples of how similar tasks were handled.

    Attributes
    ----------
    examples : List[str]
        A list of strings, each representing an example to be prepended to the primary prompt.

    Methods
    -------
    __call__(sample: PromptDict) -> PromptDict:
        Prepends the examples to the given sample's prompt, creating a new prompt with contextual examples.
    """
    examples: List[str]

    def __call__(self, sample: PromptDict) -> PromptDict:
        """
        Prepends provided examples to the given sample's prompt.

        Parameters
        ----------
        sample : PromptDict
            A dictionary containing the original prompt to which the examples will be prepended.

        Returns
        -------
        PromptDict
            A new dictionary with the 'prompt' key containing the combined text of the examples and the original prompt.
        """
        context = ""

        for example in self.examples:
            context += example + "\n\n\n" # Close the open bracket and add newlines between questions

        return { "prompt": context + sample["prompt"] }