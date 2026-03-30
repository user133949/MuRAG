# Fine-tuning prompts for answer judgment.

ANSWER_JUDGMENT_PROMPTS = """
    Please analyze the given question, standard answer, and response, and determine whether the response is correct, following these guidelines:

    Guidelines:
    - Analyze whether the response and the standard answer are semantically equivalent with respect to the question
    - Accept indirect/derived expressions (relative dates, simple arithmetic, unit conversions, aliases/synonyms) if they logically entail the same value/entity.
    - If the Question expects a single value, hedged/approximate/range/alternative claims are not equivalent unless they uniquely collapse to the Standard Answer.
    - For multi-part answers, the response is correct only if it fully matches all parts of the standard answer, any missing or conflicting part makes it incorrect.
    - If the response is fully correct and aligns with the reference answer, output: true
    - If the response is not correct, output: false
    - Do not provide explanations, reasoning, or additional text. Only output true or false.

    Input:
    - Question: {question}
    - Standard Answer: {reference_answer}
    - Response: {response}

    Judgment:
"""

ANSWER_SCORING_PROMPTS = """
    Please analyze the given question, standard answer, and response, and assign a score between 0 and 100, following these guidelines:

    Guidelines:
    - Analyze whether the response and the standard answer are semantically equivalent with respect to the question.
    - Accept indirect/derived expressions (relative dates, simple arithmetic, unit conversions, aliases/synonyms) if they logically entail the same value/entity.
    - If the Question expects a single value, hedged/approximate/range/alternative claims are not equivalent unless they uniquely collapse to the Standard Answer.
    - For multi-part answers, the response is fully correct only if it matches all parts of the standard answer; missing or conflicting parts reduce the score.
    - A fully correct response that aligns with the standard answer must receive 100.
    - A fully incorrect response must receive 0.
    - Partially correct responses should receive proportionally intermediate scores between 1 and 99.
    - Do not provide explanations, reasoning, or additional text. Only output the score as an integer.

    Input:
    - Question: {question}
    - Standard Answer: {reference_answer}
    - Response: {response}

    Score:
"""
