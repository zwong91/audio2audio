from .openai_llm import OpenAILLM

class LLMFactory:
    @staticmethod
    def create_llm_pipeline(engine_type, **kwargs):
        if engine_type == "openai":
            return OpenAILLM(**kwargs)
        else:
            raise ValueError(f"Unknown LLM pipeline type: {engine_type}")
