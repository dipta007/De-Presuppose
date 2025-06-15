SG1_VERIFIER_PROMPT_BINARY = dict(
    given_evidence="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence (i.e. corpus articles) and decide whether the available evidence would likely support or not support the claim.

Claim to evaluate:
<claim>
{CLAIM}
</claim>

Additional evidence provided:
<corpus_text>
{EVIDENCE}
</corpus_text>


Guidelines:
1. Evaluate the claim only based on the evidence provided.
2. It's possible that you are given multiple evidence articles. It is also possible that some of the evidence articles are not relevant to the claim. Use your best judgement to determine which evidence to use and which to ignore.
3. If relevant information is not present in the evidence, then it is possible that the claim is not supported by the evidence. Use your best judgement and previous knowledge to make a decision.
4. Finally, analyze the claim and evidence together and determine if the evidence supports or not supports the main claim.

After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

Now, please evaluate the above claim.
    """,
    given_evidence_questions="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence (i.e. corpus articles) and decide whether the available evidence would likely support or not support the claim. You are also given some questions that can help you analyze the claim and evidence.

Claim to evaluate:
<claim>
{CLAIM}
</claim>

Additional evidence provided:
<corpus_text>
{EVIDENCE}
</corpus_text>

Questions to consider:
<questions>
{QUESTIONS}
</questions>

Guidelines:
1. Evaluate the claim only based on the evidence provided.
2. It's possible that you are given multiple evidence articles. It is also possible that some of the evidence articles are not relevant to the claim. Use your best judgement to determine which evidence to use and which to ignore.
3. Consider answering the questions one by one, before making a final decision.
4. If relevant information is not present in the evidence, then it is possible that the claim is not supported by the evidence. Use your best judgement and previous knowledge to make a decision.

After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

Now, please evaluate the above claim.
    """,
    given_evidence_questions_answers="""\
You are an AI model tasked with verifying claims using zero-shot learning. Your job is to analyze a given claim along with provided evidence (i.e. corpus articles) and decide whether the available evidence would likely support or not support the claim. You are also given some questions and answers that can help you analyze the claim and evidence.

Claim to evaluate:
<claim>
{CLAIM}
</claim>

Additional evidence provided:
<corpus_text>
{EVIDENCE}
</corpus_text>

Questions to consider:
<questions>
{QUESTIONS}
</questions>

Answers to the questions:
<answers>
{ANSWERS}
</answers>

Guidelines:
1. Evaluate the claim only based on the evidence provided.
2. It's possible that you are given multiple evidence articles. It is also possible that some of the evidence articles are not relevant to the claim. Use your best judgement to determine which evidence to use and which to ignore.
3. Its possible that some of the questions are not relevant to the claim. Use your best judgement to determine which questions to answer and which to ignore.
4. Its possible that some of the answers are wrong. Use your best judgement to determine which answers to use and which to ignore.
5. If relevant information is not present in the evidence, then it is possible that the claim is not supported by the evidence. Use your best judgement and previous knowledge to make a decision.

After your analysis, output exactly one JSON object with exactly two keys: \"reasoning\" and \"decision\". The value associated with \"decision\" must be exactly one word – either \"SUPPORTED\" or \"NOT_SUPPORTED\" (uppercase, with no additional text). Do not add any markdown formatting, code fences, or additional text. "
"The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{\"reasoning\": \"Your brief explanation here (one or two sentences).\", \"decision\": \"SUPPORTED or NOT_SUPPORTED\"}}

Now, please evaluate the above claim.
    """,
)
