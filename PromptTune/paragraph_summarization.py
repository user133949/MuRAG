# Prompt templates for paragraph summarization

TEXT_SUMMARY_SYSTEM = "You are an expert text summarizer. Provide accurate, concise summaries of text segments."

# Text summarization prompt template
TEXT_SUMMARY_PROMPTS = """
    Please summarize the following text segment accurately and concisely, following these guidelines:

    Guidelines:
    - Maximum length: 50 words.
    - The output must begin with: "Page {page_idx}, Segment {segment_idx}: " followed by the summary.
    - Always write in one natural, flowing sentence (no bullet points, no JSON, no separate fields).
    - Focus only on the core meaning of the segment.
    - Do not add interpretation, speculation, or information not present in the text.
    - Ensure the summary captures the central idea without unnecessary detail.
    - Use neutral, clear, and precise language.

    Example:
    Input:
        Page index: 3, 
        Segment index: 2
        text = "In 2021, the company faced significant challenges due to global supply chain disruptions, 
                rising raw material costs, and shifting consumer demand. Despite these obstacles, 
                it managed to adapt quickly by diversifying suppliers, investing in local production facilities, 
                and launching several new product lines aimed at digital-savvy customers."
    Output:
        Page 3, Segment 2: The company overcame supply chain and cost challenges in 2021 by diversifying suppliers, expanding local production, and launching new digital-focused products.
        

    Segment information:
    - Page index: {page_idx}
    - Segment index: {segment_idx}
    - Text: {text}

    Task:
    - Summarize the main idea of this segment in clear, concise sentence.
"""
