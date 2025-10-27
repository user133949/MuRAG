# Prompt templates for multimodal content (images, tables, equations) processing.

# System prompts for different analysis types
IMAGE_ANALYSIS_SYSTEM = "You are an expert image analyst. Provide detailed, accurate descriptions."
TABLE_ANALYSIS_SYSTEM = "You are an expert data analyst. Provide detailed table analysis with specific insights."
EQUATION_ANALYSIS_SYSTEM = "You are an expert mathematician. Provide detailed mathematical analysis."

# Image analysis prompt template
IMAGE_ANALYSIS_PROMPTS = """
    Please analyze this image in detail and provide a comprehensive and detailed visual description, following these guidelines:

    Guidelines:
    - Maximum length: 100 words.
    - Always write in one natural, flowing paragraph (no bullet points, no JSON, no separate fields).
    - Describe the overall composition and layout
    - Describe all visible objects, people, text, and visual elements.
    - Explain relationships and interactions between elements.
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - If additional context is provided (captions or footnotes), seamlessly integrate it into the description to enrich or clarify the visual details.
    - If no context is provided, simply describe the image without mentioning the absence of context.
    - If the context provides specific names, use those names directly instead of pronouns
    - Always use specific names instead of pronouns

    Additional context:
    - Captions and Footnotes: {context}

    Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval.
"""

# Table analysis prompt template
TABLE_ANALYSIS_PROMPTS = """
    Please analyze this table content and provide a comprehensive analysis of the table, following these guidelines:

    Guidelines:
    - Maximum length: 100 words.
    - Always write in one natural, flowing paragraph (no bullet points, no JSON, no separate fields).
    - Describe the overall structure and layout of the table (rows, columns, headers).
    - Identify and explain the column headers and what they represent.
    - Describe the key data points, values, and patterns shown in the table.
    - Highlight any statistical insights, comparisons, or notable trends.
    - Explain relationships between different data elements across rows and columns.
    - If additional context is provided (captions or footnotes), seamlessly integrate it into the description to enrich or clarify the interpretation of the table.
    - If no context is provided, simply describe the table without mentioning the absence of context.
    - Always use specific names and values instead of general references.

    Additional context:
    - Captions and Footnotes: {context}

    Focus on extracting meaningful insights and relationships from the tabular data.
"""

# Equation analysis prompt template
EQUATION_ANALYSIS_PROMPTS = """
    Please analyze this mathematical equation and provide a comprehensive analysis of the equation, following these guidelines:

    Guidelines:
    - Maximum length: 100 words.
    - Always write in one natural, flowing paragraph (no bullet points, no JSON, no separate fields).
    - Explain the mathematical meaning and interpretation of the equation.
    - Identify and describe all variables and their definitions.
    - Describe the mathematical operations, functions, or structures present.
    - Explain the application domain (e.g., physics, economics, statistics) if it can be inferred.
    - Discuss the theoretical or physical significance of the equation.
    - Describe any relationships to other relevant mathematical concepts.
    - Mention practical applications or use cases if applicable.
    - If additional context is provided (Table body), seamlessly integrate it into the description to enrich or clarify the interpretation of the equation.
    - If no context is provided, simply describe the equation without mentioning the absence of context.
    - Always use precise mathematical terminology instead of vague wording.

    Focus on providing mathematical insights and explaining the equation's significance.

    Additional context:
    - Table body: {context}
"""


