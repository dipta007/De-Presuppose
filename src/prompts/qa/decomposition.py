PROMPT_FROM_WICE = """\
You are given a claim, your task is to decompose it into multiple independent and individual questions. DON'T generate any other text than the questions. You are given some examples below and the input claim at the end.

Claim: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Questions:
- Did Lord Steven Regal win the World Television Championship?
- Did The Nasty Boys win the World Tag Team Championship?

Claim: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Questions:
- When was the parkway opened?
- How long was the construction period for the parkway?
- How many years of community requests preceded the opening of the parkway?

Claim: Touring began in Europe in April–June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July–August, and concluding in North America in October–November.
Questions:
- When did touring begin in Europe?
- Who was the opening act during the touring in Europe?
- Which months covered the Australia tour?
- Which months covered the New Zealand tour?
- Which months covered the Mexico tour?
- Which months covered the South America tour?
- Which months covered the North America tour?
- Where did the touring conclude?

Claim: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Questions:
- When did the company partner with AWS?
- What was the purpose of the partnership?

Claim: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Questions:
- Which country hosts the largest Yazidi community?
- How large is the Yazidi community in Germany?
- In which cities are the Yazidi community in Germany primarily located?

Claim: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2–0.
Questions:
- How many times had Sebastian Vettel won the Nations' Cup before?
- What title did Sebastian Vettel achieve for the first time?
- Whom did Sebastian Vettel defeat in the final?
- How many finals had Tom Kristensen reached?
- What was the final score between Sebastian Vettel and Tom Kristensen?

Claim: {claim}
Questions:
"""

PQA_QUESTION2ANSWER_PROMPT = """\
Given a specific question and a relevant document, generate the most accurate and concise answer based exclusively on the information present within that document.

### Instructions:
- The answer MUST be derived solely from the text provided in the document. Do NOT infer, interpret, or add external knowledge.
- The answer MUST be accurate. Faithfully represent what the document states regarding the question.
- The answer MUST be concise. Provide the most direct answer possible. Aim for a single sentence, but include essential clarifying details from the document if strictly required for accuracy and completeness concerning the question asked.
- If the document does NOT contain the necessary information to answer the question, the output MUST be exactly: "Information not found in the provided document."
- The answer MUST be understandable on its own, without referencing the question or the structure of the document.
- Output ONLY the resulting answer sentence OR the "Information not found in the provided document." statement. Do not include any other text, labels, or explanations.

### Input:
Document:
{document}

Question:
{question}

### Output:
"""
