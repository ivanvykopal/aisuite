from aisuite.framework.choice import Choice
from aisuite.framework.message import CompletionUsage
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from vllm import LLM


class VllmProvider(Provider):
    def __init__(self):
        self.llm = None
    
    def load_llm(self, model_name: str, quanization: str = None):
        if self.llm is None:
            if quanization:
                self.llm = LLM(model_name, quantization=quanization)
            else:
                self.llm = LLM(model_name)
        
    def chat_completions_create(self, model, messages, **kwargs):
        quantization = kwargs.get("quantization", None)
        self.load_llm(model, quantization)
        
        sampling_params = self.llm.get_default_sampling_params()
        
        if "temperature" in kwargs:
            sampling_params.temperature = kwargs["temperature"]
        if "top_p" in kwargs:
            sampling_params.top_p = kwargs["top_p"]
        if "max_tokens" in kwargs:
            sampling_params.max_tokens = kwargs["max_tokens"]
        if "top_k" in kwargs:
            sampling_params.top_k = kwargs["top_k"]
            
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params
        )
        return self._normalize_response(outputs)
    
    def _normalize_response(self, response_data):        
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data[0].outputs[0].text
        normalized_response.finish_reason = response_data[0].outputs[0].finish_reason
        normalized_response.usage = CompletionUsage(
            prompt_tokens=len(response_data[0].prompt_token_ids),
            completion_tokens=len(response_data[0].outputs[0].token_ids),
        )
        return normalized_response
