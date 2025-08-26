MINICHECK_VERIFIER_PROMPT_BINARY = dict(
    given_evidence="""\
Determine whether the provided claim is supported by the corresponding document. Support in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered not supported.

Document:
{EVIDENCE}

Claim:
{CLAIM}

Please assess the claim's support with the document by responding with either "SUPPORTED" or "NOT_SUPPORTED". Do not generate anything else other than the answer.

Answer:
""",
    given_evidence_questions="""\
Determine whether the provided claim is supported by the corresponding document. You are also given some decomposed questions derived from the claim. Reason through the questions to support your judgment. Support in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered not supported. Its possible that some of the questions are not relevant to the claim. Use your best judgement to determine which questions to consider and which to ignore. Fall back to the provided document when you are not sure about the question.

Document:
{EVIDENCE}

Claim:
{CLAIM}

Questions to consider:
{QUESTIONS}

Please assess the claim's support with the document by responding with either "SUPPORTED" or "NOT_SUPPORTED". Do not generate anything else other than the answer.

Answer:
""",
    given_evidence_questions_answers="""\
Determine whether the provided claim is supported by the corresponding document. You are also given some decomposed questions derived from the claim and their answers. Reason through the questions and answers to support your judgment. Support in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered not supported. Its possible that some of the questions are not relevant to the claim. Also, its possible that some of the answers are wrong. Use your best judgement to determine which questions or answers to use and which to ignore. Fall back to the provided document when you are not sure about the question or answer.

Document:
{EVIDENCE}

Claim:
{CLAIM}

Questions to consider:
{QUESTIONS}

Answers to the questions:
{ANSWERS}

Please assess the claim's support with the document by responding with either "SUPPORTED" or "NOT_SUPPORTED". Do not generate anything else other than the answer.

Answer:
""",
)
