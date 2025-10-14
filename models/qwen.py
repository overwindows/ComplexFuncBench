import os
import re
import copy
import json
from typing import Any, Dict
from utils.utils import *
from openai import OpenAI

"""
You can also deploy Qwen2.5 via vLLM, please enable the auto-tool-choice. 
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```
Note: Tool support has been available in vllm since v0.6.0. Be sure to install a version that supports tool use.
Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm
"""

class QwenModel:
    def __init__(self, url, model_name):
        self.model_name = model_name
        self.messages = []
        self.url = url
        self.client = OpenAI(
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url=os.getenv("SAMBANOVA_API_URL"))

    def parse_xml_tool_calls(self, xml_content):
        """
        Parse XML-style tool calls from SambaNova's error output.
        Supports multiple tool calls in format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        # Find all tool_call tags (support multiple calls)
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, xml_content, re.DOTALL)
        
        if not matches:
            return None
        
        tool_calls = []
        for idx, match in enumerate(matches):
            try:
                # Parse JSON from tool_call content
                tool_data = json.loads(match.strip())
                
                # Create a mock tool call object that mimics OpenAI's structure
                class MockToolCall:
                    def __init__(self, name, arguments, call_id):
                        self.id = call_id
                        self.type = 'function'
                        self.function = type('obj', (object,), {
                            'name': name,
                            'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
                        })()
                
                # Use consistent ID generation
                call_id = f'call_xml_{abs(hash(match)) % 10000000000}_{idx}'
                tool_calls.append(MockToolCall(
                    tool_data['name'], 
                    tool_data.get('arguments', {}),
                    call_id
                ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call JSON: {match[:100]}... Error: {e}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error parsing tool call: {e}")
                continue
        
        return tool_calls if tool_calls else None

    #@retry(max_attempts=5, delay=20)
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
            # Return the message object directly like DeepSeek
            return completion.choices[0].message
            
        except Exception as e:
            # Handle SambaNova's Qwen XML format issue
            error_str = str(e)
            
            # Check if this is the XML tool call extraction error
            if 'error_model_output' in error_str and '<tool_call>' in error_str:
                print(f"\n[Qwen XML Parser] Detected XML-style tool call in error, parsing...")
                
                try:
                    # Extract error_model_output from the error string
                    # The error format is: {'error': '...', 'error_model_output': '...', ...}
                    match = re.search(r"'error_model_output':\s*'(.*?)'(?:,\s*'error_param'|$)", error_str, re.DOTALL)
                    if match:
                        model_output = match.group(1)
                        # Unescape the string
                        model_output = model_output.replace("\\'", "'").replace("\\n", "\n")
                        
                        # Try to parse tool calls from XML
                        parsed_calls = self.parse_xml_tool_calls(model_output)
                        if parsed_calls:
                            print(f"[Qwen XML Parser] Successfully parsed {len(parsed_calls)} tool call(s)")
                            
                            # Create a mock message object compatible with OpenAI format
                            class MockMessage:
                                def __init__(self, tool_calls):
                                    self.role = 'assistant'
                                    self.content = None
                                    self.tool_calls = tool_calls
                                    self.refusal = None
                                    self.function_call = None
                            
                            return MockMessage(parsed_calls)
                        else:
                            print(f"[Qwen XML Parser] Failed to parse tool calls from XML")
                            print(f"[Qwen XML Parser] Output snippet: {model_output[:300]}")
                            
                except Exception as parse_err:
                    import traceback
                    print(f"[Qwen XML Parser] Error during XML parsing: {parse_err}")
                    print(traceback.format_exc())
            
            # If XML parsing failed or it's a different error, log and return None
            print(f"\n{'='*80}")
            print(f"[Qwen Error] API call failed: {str(e)[:200]}")
            print(f"{'='*80}\n")
            return None