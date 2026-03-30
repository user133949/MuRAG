# Fine-tuning prompts for domain generation.

GENERATE_DOMAIN_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by assigning a descriptive domain that summarizes what the text is about.
Example domains are: "Social studies", "Algorithmic analysis", "Medical science", among others.
Only output your final answer in the string, and do NOT include any other text, thoughts, or tags.

Text: {input_text}
Domain:"""
