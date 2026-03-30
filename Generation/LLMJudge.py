import logging
from PromptTune.answer_judgement import ANSWER_JUDGMENT_PROMPTS, ANSWER_SCORING_PROMPTS
from LLM.LLMclient import ChatModel

def judge_answer(model: str,
                 reasoning_model: str,
                 question: str,
                 reference_answer: str,
                 response: str,
                 api_key="ollama", 
                 base_url="http://localhost:11434/v1/"
                 ):
    """使用大模型对答案进行判断"""
    chatLLM = ChatModel(model=model,
                        reasoning_model=reasoning_model,
                        api_key=api_key,
                        base_url=base_url)
    try:
        results = chatLLM.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_JUDGMENT_PROMPTS.format(question=question, reference_answer=reference_answer, response=response) + "/no_think",
                }, 
                {
                    "role": "system", 
                    "content": "you are a helpful AI assistant that can judge whether the response is correct or not."
                }
            ],
            temperature=0.0,
            max_tokens=256,
        )
        result = results.choices[0].message.content
    except Exception as e:
        result = "Failed"
        logging.error(f"Failed to judge the answer, an error occurred in LLM: {e}")
    
    return result

def score_answer(model: str, 
                 reasoning_model: str, 
                 question: str, 
                 reference_answer: str, 
                 response: str, 
                 api_key="ollama", 
                 base_url="http://localhost:11434/v1/"
                 ):
    """使用大模型对答案进行评分"""
    chatLLM = ChatModel(model=model,
                        reasoning_model=reasoning_model,
                        api_key=api_key,
                        base_url=base_url)
    try:
        results = chatLLM.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_SCORING_PROMPTS.format(question=question, reference_answer=reference_answer, response=response) + "/no_think",
                }, 
                {
                    "role": "system", 
                    "content": "you are a helpful AI assistant that can score the response based on the question and reference answer."
                }
            ],
            temperature=0.0,
            max_tokens=256,
        )
        result = results.choices[0].message.content
    except Exception as e:
        result = "Failed"
        logging.error(f"Failed to judge the answer, an error occurred in LLM: {e}")

    return result
