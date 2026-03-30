import json
from LLM.LLMclient import ChatModel
from EntityExtraction.defaults import DEFAULT_TASK
from PromptTune.entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    ENTITY_TYPE_GENERATION_PROMPT
    )

def generate_entity_types(chat_model: ChatModel, 
                          domain: str, 
                          persona: str, 
                          docs: str | list[str], 
                          task: str = DEFAULT_TASK, 
                          json_mode: bool = False
                          ) -> str | list[str]:
    """
    Generate entity type categories from a given set of (small) documents.

    Example Output:
    json mode: ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']
    not json mode: 'military unit, organization, person, location, event, date, equipment'
    
    json mode is not recommended !
    """
    formatted_task = task.format(domain=domain)
    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    entity_types_prompt = (ENTITY_TYPE_GENERATION_JSON_PROMPT 
                           if json_mode else ENTITY_TYPE_GENERATION_PROMPT
                           ).format(task=formatted_task, input_text=docs_str)

    if json_mode:
        response = chat_model.chat_with_llm(prompt=entity_types_prompt, 
                                            system_content=persona)
        try:
            response = json.loads(response)
            response = response.get("entity_types")
        except Exception as e:
            print(f"Unexpected output, JSON format error: {e}")
    else:
        response = chat_model.chat_with_llm(prompt=entity_types_prompt, 
                                            system_content=persona)
    
    return response