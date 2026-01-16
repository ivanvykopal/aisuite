from aisuite.utils.config import Config
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
                
    def _normalize_parameters(self, **kwargs):
        # kwargs should have only attributes defined in Config if there is any extra attribute, it will raise an error, all missing should be filled with default value
        config = Config(**kwargs)
        
        return {
            "quantization": config.quantization,
            "n": config.n,
            "best_of": config.best_of,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "repetition_penalty": config.repetition_penalty,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "seed": config.seed,
            "stop": config.stop,
            "max_tokens": config.max_completion_tokens,
            "min_tokens": config.min_tokens,
            "skip_special_tokens": config.skip_special_tokens,
            "logit_bias": config.logit_bias,
        }
        
    def chat_completions_create(self, model, messages, **kwargs):
        params = self._normalize_parameters(**kwargs)
        self.load_llm(model, params.pop("quantization"))
        
        sampling_params = self.llm.get_default_sampling_params()
        for key, value in params.items():
            if value is not None:
                # sampling param is not dict but object with attributes
                setattr(sampling_params, key, value)
            
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
