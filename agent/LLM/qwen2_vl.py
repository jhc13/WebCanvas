from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)


def process_messages(messages: list[dict]) -> list[dict]:
    """Convert messages into the format expected by Qwen2-VL."""
    processed_messages = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            processed_content = [{'type': 'text', 'text': content}]
        else:
            processed_content = []
            for content_part in content:
                if content_part['type'] == 'image_url':
                    base64_image = content_part['image_url']['url']
                    processed_content.append({'type': 'image',
                                              'image': base64_image})
                else:
                    processed_content.append(content_part)
        processed_message = {'role': message['role'],
                             'content': processed_content}
        processed_messages.append(processed_message)
    return processed_messages


class Qwen2VlGenerator:
    _instance = None
    model_ = None

    def __new__(cls, *args, **kwargs):
        # If a previous instance exists, return it instead of creating a new
        # one.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_id: str):
        # Do not load the model if it has already been loaded.
        if self.model_:
            return
        self.model = model_id
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_ = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation='flash_attention_2',
            device_map='cuda', quantization_config=quantization_config)
        self.processor = AutoProcessor.from_pretrained(model_id)

    async def request(self, messages: list, max_tokens: int = 500,
                      temperature: float = 0.7) -> tuple[str, str]:
        messages = process_messages(messages)
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response_start = '{\n    "thought":'
        text_input += response_start
        image_input, video_input = process_vision_info(messages)
        inputs = self.processor(text=[text_input], images=image_input,
                                videos=video_input, padding=True,
                                return_tensors='pt')
        inputs = inputs.to('cuda')
        input_token_count = inputs.input_ids.shape[1]
        generated_token_ids = self.model_.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature)[0][input_token_count:]
        generated_text = self.processor.decode(
            generated_token_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        generated_text = response_start + generated_text
        return generated_text, ''
