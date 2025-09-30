from typing import Any
import os
import sys
import copy
import json
from openai import OpenAI

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.prompts import SimpleTemplatePrompt  # noqa: E402
from utils.utils import retry  # noqa: E402


class DeepSeekModel:
    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__()
        self.model_name = model_name
        # DeepSeek API configuration
        self.client = OpenAI(
            api_key="042ca35c-beaf-4f5b-8033-9170556e5251",
            base_url="https://api.sambanova.ai/v1"
        )

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction

    #@retry(max_attempts=1)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


class FunctionCallDeepSeek(DeepSeekModel):
    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__(model_name, api_key, base_url)
        self.messages = []

    #@retry(max_attempts=4, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    # Test basic model
    model = DeepSeekModel("deepseek-chat")
    response = model(
        "You are a helpful assistant.",
        SimpleTemplatePrompt(
            template=("What is the capital of France?"),
            args_order=[]
        )
    )
    print("Basic model response:", response)

    # Test function calling model
    func_model = FunctionCallDeepSeek("deepseek-chat")
    test_messages = [
        {"role": "user", "content": "What's the weather like?"}
    ]
    func_response = func_model(test_messages)
    print("Function call model response:", func_response)
