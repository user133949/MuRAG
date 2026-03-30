# Fine-tuning prompts for language detection.

DETECT_LANGUAGE_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by determining what's the primary language of the provided texts.
Examples are: "English", "Spanish", "Japanese", "Portuguese" among others. Reply ONLY with the language name.
Only output your final answer in the string, and do NOT include any other text, thoughts, or tags.

Text: {input_text}
Language:"""
