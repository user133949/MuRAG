from LLM.LLMclient import ChatModel
from PromptTune.language_detection import DETECT_LANGUAGE_PROMPT

def detect_language(chat_model: ChatModel, 
                    docs: str | list[str]
                    ) -> str:
    """Detect input language to use for GraphRAG prompts.

    Parameters
    ----------
    - docs (str | list[str]): The docs to detect language from
    - model (CompletionLLM): The LLM to use for generation

    Returns
    -------
    - str: The detected language.
    """
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    language_prompt = DETECT_LANGUAGE_PROMPT.format(input_text=docs_str)

    response = chat_model.chat_with_llm(prompt=language_prompt, 
                                        system_content="You are an intelligent assistant that helps a human to analyze the information in a text document."
                                        )

    return str(response)
