import copy
import json
import os
import sys
from typing import Any

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *


# Define recursive function to process format fields
def process_tool_format_fields(schema_obj):
    if not isinstance(schema_obj, dict):
        return

    # Process format at current level
    if "format" in schema_obj and "description" in schema_obj:
        format_value = schema_obj.pop("format")
        existing_desc = schema_obj.get("description", "")
        separator = " " if existing_desc else ""
        schema_obj["description"] = (
            existing_desc + separator + f"Use the following format: {format_value}"
        )

    # Process properties at current level
    if "properties" in schema_obj and isinstance(schema_obj["properties"], dict):
        for prop_name, prop_config in schema_obj["properties"].items():
            process_tool_format_fields(prop_config)

    # Process items for array types
    if "items" in schema_obj and isinstance(schema_obj["items"], dict):
        process_tool_format_fields(schema_obj["items"])


def add_root_type_if_missing(schema: dict):
    if "type" not in schema:
        schema["type"] = "object"


def recursively_set_additional_properties_false(schema: dict):
    if not isinstance(schema, dict):
        return
    if (
        "additionalProperties" not in schema or schema["additionalProperties"]
    ) and schema.get("properties"):
        schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


class GPTModel:
    def __init__(self, model_name, api_key=None, base_url=None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction

    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


class FunctionCallGPT(GPTModel):
    def __init__(self, model_name, api_key=None, base_url=None, **kwargs):
        super().__init__(model_name, api_key, base_url)
        self.messages = []
        self.strict = kwargs.get("strict", True)

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)

        if tools is not None:
            for tool in tools:
                process_tool_format_fields(tool["function"]["parameters"])
                add_root_type_if_missing(tool["function"]["parameters"])
                recursively_set_additional_properties_false(
                    tool["function"]["parameters"]
                )
                tool["function"]["strict"] = self.strict

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048,
            )
            return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    model = GPTModel("gpt-4")
    response = model(
        "You are a helpful assistant.",
        SimpleTemplatePrompt(
            template=("What is the capital of France?"), args_order=[]
        ),
    )
    print(response)
