SG2_VERIFIER_PROMPT_BINARY = dict(
    given_evidence="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence and decide whether the available evidence would likely support or not support the claim.

### Instructions:
1. Evaluate the claim only based on the evidence provided.
2. Finally, analyze the claim and evidence together and determine the label that best describes the relationship between the claim and the evidence.
3. The meaning of different labels:
    - SUPPORTED: The claim is supported by the evidence.
    - NOT_SUPPORTED: The claim is not supported by the evidence.

### Output Format:
After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

### Input:
Evidence:
{EVIDENCE}

Claim to evaluate:
{CLAIM}

### Output:
""",
    given_evidence_questions="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence and decide whether the available evidence would likely support or not support the claim. You are also given some questions that can help you analyze the claim and evidence.

### Instructions:
1. Evaluate the claim only based on the evidence provided.
2. Consider answering the questions one by one, before making a final decision.
3. It is possible that some of the questions are not relevant to the claim. Use your best judgement to determine which questions to answer and which to ignore.
4. Finally, analyze the claim, questions and evidence together and determine the label that best describes the relationship between the claim and the evidence.
5. The meaning of different labels:
    - SUPPORTED: The claim is supported by the evidence.
    - NOT_SUPPORTED: The claim is not supported by the evidence.

### Output Format:
After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

### Input:
Evidence:
{EVIDENCE}

Claim to evaluate:
{CLAIM}

Questions to consider:
{QUESTIONS}

### Output:
""",
    given_evidence_questions_answers="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence and decide whether the available evidence would likely support or not support the claim. You are also given some questions and their answers that can help you analyze the claim and evidence.

### Instructions:
1. Evaluate the claim only based on the evidence provided.
2. It is possible that some of the questions are not relevant to the claim. Use your best judgement to determine which questions to answer and which to ignore.
3. It is possible that some of the answers are wrong. Use your best judgement to determine which answers to use and which to ignore.
4. Finally, analyze the claim, questions, answers and evidence together and determine the label that best describes the relationship between the claim and the evidence.
5. The meaning of different labels:
    - SUPPORTED: The claim is supported by the evidence.
    - NOT_SUPPORTED: The claim is not supported by the evidence.

### Output Format:
After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

### Input:
Evidence:
{EVIDENCE}

Claim to evaluate:
{CLAIM}

Questions to consider:
{QUESTIONS}

Answers to the questions:
{ANSWERS}

### Output:
""",
)
