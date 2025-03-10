# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load.dump import dumps
from langchain_core.messages import ChatMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
from typing import List, Mapping, Optional, Any
import requests
import json
import logging

_LOGGER = logging.getLogger(__name__)

class OllamaChatModel(BaseChatModel):
    """A LangChain chat model for Ollama API."""

    ollama_server: str = Field("http://localhost", description='URL of the Ollama server')
    ollama_port: str = Field("11434", description='Port of the Ollama server')
    model_name: str = Field("llama3", description='Name of the Ollama model to use')
    temperature: float = Field(0.7, description='Temperature for text generation')
    
    def __init__(self, ollama_server="http://localhost", ollama_port="11434", model_name="llama3", temperature=0.7, **kwargs):
        super().__init__(**kwargs)
        self.ollama_server = ollama_server
        self.ollama_port = ollama_port
        self.model_name = model_name
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return 'ollama'
        
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self._call_ollama_api(messages)
        return self._create_chat_result(response)
    
    def _call_ollama_api(self, messages, **kwargs):
        """Call the Ollama API to generate text."""
        base_url = f"{self.ollama_server}:{self.ollama_port}/api/chat"
        
        # Convert LangChain messages to Ollama format
        obj = json.loads(dumps(messages))
        prompt_content = obj[0]["kwargs"]["content"]
        
        # Format for Ollama chat API
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "options": {
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(base_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            _LOGGER.error(f"Error calling Ollama API: {e}")
            return {"message": {"content": f"Error: Could not connect to Ollama server at {self.ollama_server}:{self.ollama_port}"}}
    
    def _create_chat_result(self, response):
        """Create a ChatResult from the Ollama response."""
        try:
            content = response.get("message", {}).get("content", "No response from model")
            message = ChatMessage(content=content, role="assistant")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            _LOGGER.error(f"Error creating chat result: {e}")
            message = ChatMessage(content=f"Error processing model response: {str(e)}", role="assistant")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    def list_models(self):
        """List all available models in the Ollama server."""
        url = f"{self.ollama_server}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            _LOGGER.error(f"Error listing Ollama models: {e}")
            return []
            
    def pull_model(self, model_name):
        """Pull a model from the Ollama library if not already installed."""
        url = f"{self.ollama_server}:{self.ollama_port}/api/pull"
        payload = {
            "name": model_name
        }
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            _LOGGER.error(f"Error pulling Ollama model {model_name}: {e}")
            return False