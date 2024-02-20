from typing import TypedDict


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


class CommonSenseQAPrompt:
    
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
        choices = f"{CHOICES_KEY} \n" + "".join(
            [
                f"({label.lower()}) {text.lower()}\n"
                for label, text in zip(
                    sample["choices"]["label"],
                    sample["choices"]["text"]
                )
            ]
        )
        answer = f"{ANSWER_KEY} ({sample['answerKey'].lower()})"

        prompt = f"{question}\n{choices}\n{answer}"

        return {'prompt': prompt}