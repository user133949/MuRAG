# Fine-tuning prompts for answer generation.

MULTIMODAL_GENERATE_ANSWER_PROMPT= """
You are an expert assistant that answers questions using text chunks, image descriptions, and images.

User Question: {query}

Text Chunks Information: {text_chunks}

Image Description Information: {image_description}

Instructions:
1. Carefully read the question.
2. Analyze both the text information and the provided images.
3. Image description information may include: caption and footnote of image, LaTeX source code of equation, HTML table code.
4. Use the image description information to understand the images.
5. Combine insights from the text and the visual information to answer the question.
6. If the question cannot be answered based on the given information, and you do not know the answer, respond with "Not answerable".
6. Provide the final answer in a concise, clear, and well-structured manner.
7. Cite or reference information from the text and images where appropriate.

Answer:
"""

UNIMODAL_GENERATE_ANSWER_PROMPT= """
You are an expert assistant that answers questions using only text chunks.

User Question: {query}

Text Chunks Information: {text_chunks}

Instructions:
1. Carefully read the question.
2. Analyze the provided text information.
3. Extract key details and relevant evidence from the text chunks.
4. If the question cannot be answered based on the given text chunks, and you do not know the answer, respond with "Not answerable".
5. Provide the final answer in a concise, clear, and well-structured manner.
6. Cite or reference the supporting text information where appropriate.

Answer:
"""

VLM_GENERATE_ANSWER_PROMPT= """
You are an expert assistant that answers questions using images.

User Question: {query}

Instructions:
1. Carefully read the question.
2. Analyze the provided images.
3. If the question cannot be answered based on the given information, and you do not know the answer, respond with "Not answerable".
4. Provide the final answer in a concise, clear, and well-structured manner.
5. Cite or reference information from the images where appropriate.

Answer:
"""