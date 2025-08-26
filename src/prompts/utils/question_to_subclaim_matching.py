QUESTION_TO_SUBCLAIM_MATCHING_PROMPT = """\
Given a question and a list of subclaims, determine which subclaims are relevant to the question.

### Instructions
- The question does not need to ask the specific claim explicitly. If the subclaim even partially answers the question, then it is relevant to the question.
- It is possible that multiple subclaims are relevant to the question. It is not necessary that the question is covered by a single subclaim.
- Begin by providing 1-2 sentences explaining your reasoning for the relevance of the subclaims to the question.
- Afterward, output the subclaim ids that are relevant to the question in a python list. DO NOT include any other text in your response, so that the response can be parsed as a python list.
- Structure your final response into two sections:
    - EXPLANATION: (your reasoning in 1-2 sentences)
    - SUBCLAIM_IDS: (the subclaim ids that are relevant to the question)

### Question
{question}

### Subclaims
{subclaims}
"""


def get_question_to_subclaim_matching_to_answer(questions, subclaims):
    answers = []
    for question in questions:
        subclaims_str = "\n".join([f"{i + 1}. {subclaim}" for i, subclaim in enumerate(subclaims)])
        answer = QUESTION_TO_SUBCLAIM_MATCHING_PROMPT.format(question=question, subclaims=subclaims_str)
        answers.append(answer)
    return answers
