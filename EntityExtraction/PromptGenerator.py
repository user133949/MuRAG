from pathlib import Path
from EntityExtraction.Tokens import num_tokens_from_string
from LLM.LLMclient import ChatModel
from PromptTune.entity_relation_example import (
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
    )
from PromptTune.entity_extraction import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    GRAPH_EXTRACTION_JSON_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
    UNTYPED_GRAPH_EXTRACTION_JSON_PROMPT
    )
from EntityExtraction.defaults import (
    DEFAULT_LANGUAGE, 
    ENCODING_MODEL, 
    MAX_EXAMPLES,
    EXTRACT_GRAPH_FILENAME
    )

def generate_entity_relationship_examples(
        chat_model: ChatModel,
        entity_types: str | list[str] | None,
        docs: str | list[str],
        language: str | None,
        json_mode: bool = True,
        persona="You are a helpful assistant."
        ) -> list[str]:
    """Generate a list of entity/relationships examples for use in generating an entity configuration.

    Will return entity/relationships examples as either JSON or in tuple_delimiter format depending
    on the json_mode parameter.
    """
    docs_list = [docs] if isinstance(docs, str) else docs
    language = language if language else DEFAULT_LANGUAGE

    if entity_types:
        entity_types_str = (
            entity_types
            if isinstance(entity_types, str)
            else ", ".join(map(str, entity_types))
        )

        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            (
                UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(input_text=doc, language=language)
            for doc in docs_list
        ]
    messages = messages[:MAX_EXAMPLES]
    results = []
    for message in messages:
        response = chat_model.chat_with_llm(prompt=message, 
                                            system_content=persona)
        results.append(response)
    return results

def create_extract_graph_prompt(
    entity_types: str | list[str] | None,
    docs: list[str],
    examples: list[str],
    language: str | None,
    max_token_count: int,
    model: str,
    encoding_model: str = ENCODING_MODEL,
    json_mode: bool = False,
    output_path: Path | None = None,
    min_examples_required: int = 2, 
    max_examples_allowed: int = MAX_EXAMPLES
) -> str:
    """
    Create a prompt for entity extraction.

    Parameters
    ----------
    - entity_types (str | list[str]): The entity types to extract
    - docs (list[str]): The list of documents to extract entities from
    - examples (list[str]): The list of examples to use for entity extraction
    - language (str): The language of the inputs and outputs
    - model (str): The name of the model to use for entity extraction
    - encoding_model (str): The name of the model to use for token counting
    - max_token_count (int): The maximum number of tokens to use for the prompt
    - json_mode (bool): Whether to use JSON mode for the prompt. Default is False
    - output_path (Path | None): The path to write the prompt to. Default is None.
        - min_examples_required (int): The minimum number of examples required. Default is 2.

    Returns
    -------
    - str: The entity extraction prompt
    """
    
    prompt = (
        GRAPH_EXTRACTION_JSON_PROMPT if entity_types is not None and json_mode else
        GRAPH_EXTRACTION_PROMPT if entity_types is not None else
        UNTYPED_GRAPH_EXTRACTION_JSON_PROMPT if json_mode else
        UNTYPED_GRAPH_EXTRACTION_PROMPT
        )
    
    if isinstance(entity_types, list):
        entity_types = ", ".join(map(str, entity_types))

    tokens_left = (
        max_token_count
        - num_tokens_from_string(prompt, model, encoding_name=encoding_model)
        - num_tokens_from_string(entity_types, model, encoding_name=encoding_model)
        if entity_types
        else 0  # 0使得第一个例子后的其他例子不被添加
    )

    examples_prompt = ""

    # Iterate over examples, while we have tokens left or examples left
    for i, output in enumerate(examples):
        if i >= max_examples_allowed:
            break
        input = docs[i]
        example_formatted = (
            EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, entity_types=entity_types, output=output
            )
            if entity_types
            else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, output=output
            )
        )

        example_tokens = num_tokens_from_string(example_formatted, model, encoding_name=encoding_model)

        # Ensure at least three examples are included
        if i >= min_examples_required and example_tokens > tokens_left:
            break

        examples_prompt += example_formatted
        tokens_left -= example_tokens

    language = language if language else DEFAULT_LANGUAGE

    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / EXTRACT_GRAPH_FILENAME
        # Write file to output path
        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt

def prompt_concatenate(prompt: str, 
                       docs: str | list[str], 
                       ) -> str:
    """
    拼接提示词模板与真实文本信息
    """
    docs_str = "\n".join(docs) if isinstance(docs, list) else docs
    final_prompt = prompt.replace("{input_text}", docs_str)

    return final_prompt