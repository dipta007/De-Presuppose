DEPRESUPPOSITION_PROMPT = """\
You are given a question that may contain presuppositions — assumptions that are implied but not necessarily true. Your task it to rewrite the question into one or more simpler decontextualized questions that do not contain these presuppositions. DO NOT generate anything else other than the questions. You are also given some examples below and the input question at the end.

Question:
Which bollywood movie has won the oscar in 2024?

Rewritten questions:
- Was there an oscar in 2024?
- If there was an oscar in 2024, has any bollywood movie won that?
- If any bollywood movie won the oscar in 2024, which one?

Question:
Which english movie was directed by Christopher Nolan?

Rewritten questions:
- Is Christopher Nolan a director?
- Has Christopher Nolan directed any english movie?
- If Christopher Nolan has directed any english movie, which one?

Question:
{question}

Rewritten questions:
"""
