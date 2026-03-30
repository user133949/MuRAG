"""Utilities for working with tokens."""

import logging
import tiktoken
from transformers import AutoTokenizer
import EntityExtraction.defaults as defs

DEFAULT_ENCODING_NAME = defs.ENCODING_MODEL

log = logging.getLogger(__name__)

def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    if model is not None:
        # OpenAI 系模型
        if any(key in model for key in ["gpt", "text-davinci", "code-davinci", "openai"]):
            try:
                encoding = tiktoken.encoding_for_model(encoding_name)
            except KeyError:
                msg = f"Failed to get encoding for {encoding_name} when getting num_tokens_from_string. Fall back to default encoding {DEFAULT_ENCODING_NAME}"
                log.warning(msg)
                encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
        # HuggingFace 系列模型
        else:
            try:
                encoding = AutoTokenizer.from_pretrained(encoding_name)
            except Exception as e:
                msg = f"Failed to get encoding for {encoding_name} when getting num_tokens_from_string."
                log.warning(msg)
    else:
        encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
    return len(encoding.encode(string))

def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    if model is not None:
        encoding = tiktoken.encoding_for_model(model)
    elif encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)
    return encoding.decode(tokens)
