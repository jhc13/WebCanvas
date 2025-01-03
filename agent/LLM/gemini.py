import os
import sys
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from sanic.log import logger
import google.generativeai as genai


def process_messages(messages: list[dict]) -> list[dict]:
    """
    Convert messages into the format expected by Gemini. The input messages are
    a list of dictionaries with keys `role` and `content`. The processed
    messages are a list of dictionaries with keys `role` and `parts`.
    """
    processed_messages = []
    for message in messages:
        role = message['role']
        # The `system` role is not supported.
        if role == 'system':
            role = 'user'
        content = message['content']
        if isinstance(content, str):
            parts = [content]
        else:
            # `content` is a list of dictionaries with keys `type` and `text`.
            parts = []
            for content_part in content:
                if content_part['type'] == 'text':
                    parts.append(content_part['text'])
                elif content_part['type'] == 'image_url':
                    base64_image = (content_part['image_url']['url']
                                    .split(',')[1])
                    parts.append({'mime_type': 'image/jpeg',
                                  'data': base64_image})
        processed_message = {'role': role, 'parts': parts}
        processed_messages.append(processed_message)
    return processed_messages


class GeminiGenerator:
    def __init__(self, model=None, is_json_mode: bool = True):
        self.model = model
        self.is_json_mode = is_json_mode
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    async def request(self, messages: list = None, max_tokens: int = 500, temperature: float = 0.7) -> (str, str):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(self.pool, partial(self.chat, messages, max_tokens, temperature))
            return response, ""
        except Exception as e:
            logger.error(f"Error in GeminiGenerator.request: {e}")
            return "", str(e)

    def chat(self, messages, max_tokens=500, temperature=0.7):
        chat_history = process_messages(messages)
        last_message = chat_history.pop()
        running_model = genai.GenerativeModel(self.model)
        chat = running_model.start_chat(history=chat_history)
        if self.is_json_mode:
            response_mime_type_dict = {
                'response_mime_type': 'application/json'}
        else:
            response_mime_type_dict = {}
        response = chat.send_message(
            last_message, generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature, **response_mime_type_dict))
        return response.text
