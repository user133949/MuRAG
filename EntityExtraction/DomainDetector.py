from LLM.LLMclient import ChatModel
from PromptTune.domain import GENERATE_DOMAIN_PROMPT

def generate_domain(chat_model: ChatModel, 
                    docs: str | list[str]
                    ) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - docs (str | list[str]): The domain to generate a persona for
    - model (CompletionLLM): The LLM to use for generation

    Returns
    -------
    - str: The generated domain prompt response.
    """
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    domain_prompt = GENERATE_DOMAIN_PROMPT.format(input_text=docs_str)

    response = chat_model.chat_with_llm(prompt=domain_prompt, 
                                        system_content="You are an intelligent assistant that helps a human to analyze the information in a text document."
                                        )

    return str(response)
