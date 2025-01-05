from .openai import GPTGenerator, GPTGeneratorWithJSON
from .claude import ClaudeGenerator
from .gemini import GeminiGenerator
from .qwen2_vl import Qwen2VlGenerator
from .togetherai import TogetherAIGenerator


def create_llm_instance(model, json_mode=False, all_json_models=None):
    if "gpt" in model or "o1" in model:
        if json_mode:
            if model in all_json_models:
                return GPTGeneratorWithJSON(model)
            else:
                raise ValueError("The text model does not support JSON mode.")
        else:
            return GPTGenerator(model)
    elif "claude" in model:
        if json_mode:
            raise ValueError("Claude does not support JSON mode.")
        else:
            return ClaudeGenerator(model)
    elif "gemini" in model:
        return GeminiGenerator(model, json_mode)
    elif "qwen2-vl" in model.lower():
        if json_mode:
            raise ValueError("Qwen2-vL does not support JSON mode.")
        else:
            return Qwen2VlGenerator(model)
    else:
        if json_mode:
            raise ValueError("TogetherAI does not support JSON mode.")
        else:
            return TogetherAIGenerator(model)

async def semantic_match_llm_request(messages: list = None):
    GPT35 = GPTGenerator(model="gpt-3.5-turbo")
    return await GPT35.request(messages)