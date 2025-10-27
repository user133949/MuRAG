from LLM.LLMclient import ChatModel
from EntityExtraction.defaults import DEFAULT_TASK
from PromptTune.persona import GENERATE_PERSONA_PROMPT

def generate_persona(chat_model: ChatModel,
                     domain: str, 
                     task: str = DEFAULT_TASK
                     ) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Example Output: 
    'You are an expert in community analysis. You are skilled at 
    understanding social dynamics and network structures. You are 
    adept at helping people with mapping relationships and identifying 
    key players within technology and innovation communities.'

    Parameters
    ----------
    - model (CompletionLLM): The LLM to use for generation
    - domain (str): The domain to generate a persona for
    - task (str): The task to generate a persona for. Default is DEFAULT_TASK
    """
    formatted_task = task.format(domain=domain)
    persona_prompt = GENERATE_PERSONA_PROMPT.format(sample_task=formatted_task)

    response = chat_model.chat_with_llm(prompt=persona_prompt, 
                                        system_content="You are an intelligent assistant that helps a human to analyze the information in a text document."
                                        )

    return str(response)
