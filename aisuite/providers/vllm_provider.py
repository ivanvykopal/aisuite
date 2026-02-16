from aisuite.utils.config import Config
from aisuite.framework.message import CompletionUsage
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from vllm import LLM


class VllmProvider(Provider):
    def __init__(self):
        self.llm = None

    def load_llm(
        self, model_name: str, quantization: str = None, max_model_len: int = None
    ):
        if self.llm is None:
            kwargs = {}
            if quantization:
                kwargs["quantization"] = quantization
            if max_model_len:
                kwargs["max_model_len"] = max_model_len

            self.llm = LLM(model_name, **kwargs)

    def _normalize_parameters(self, **kwargs):
        config = Config(**kwargs)

        return {
            "quantization": config.quantization,
            "max_model_len": config.max_model_len,
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
            "enable_thinking": (
                False if config.reasoning_effort in [None, "none"] else True
            ),
        }

    def _chat(self, model, messages, **kwargs):
        params = self._normalize_parameters(**kwargs)
        self.load_llm(model, params.pop("quantization"), params.pop("max_model_len"))

        sampling_params = self.llm.get_default_sampling_params()
        for key, value in params.items():
            if value is not None:
                setattr(sampling_params, key, value)

        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": params["enable_thinking"]},
        )
        return outputs

    def chat_completions_create(self, model, messages, **kwargs):
        outputs = self._chat(model, messages, **kwargs)
        return self._normalize_response(outputs)

    def batches_create(self, model, conversations, **kwargs):
        outputs = self._chat(model, conversations, **kwargs)

        normalized_responses = []
        for output in outputs:
            normalized_responses.append(self._normalize_response([output]))
        return normalized_responses

    def _normalize_response(self, response_data):
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = (
            response_data[0].outputs[0].text
        )
        normalized_response.finish_reason = response_data[0].outputs[0].finish_reason
        normalized_response.usage = CompletionUsage(
            prompt_tokens=len(response_data[0].prompt_token_ids),
            completion_tokens=len(response_data[0].outputs[0].token_ids),
        )
        return normalized_response
