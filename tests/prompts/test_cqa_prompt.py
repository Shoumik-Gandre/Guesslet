from guesslet.prompts.cqa_prompt import CommonSenseQAPrompt


def test_commonsense_qa_prompt():
    prompter = CommonSenseQAPrompt()

    example = {
        'id': '075e483d21c29a511267ef62bedc0461',
        'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
        'question_concept': 'punishing',
        'choices': {
            'label': ['A', 'B', 'C', 'D', 'E'],
            'text': [
                'ignore', 
                'enforce', 
                'authoritarian', 
                'yell at', 
                'avoid'
            ]
        },
        'answerKey': 'A'
    }
    
    correct = """Q: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
Answer Choices: 
(a) ignore
(b) enforce
(c) authoritarian
(d) yell at
(e) avoid

A: (a)"""
    output = prompter(sample=example)

    assert {'prompt': correct} == output
