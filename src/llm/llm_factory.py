from .openai_llm import OpenAILLM

class LLMFactory:
    @staticmethod
    def create_opanai_pipeline(engine_type, **kwargs):
        if engine_type == "openai":
            return OpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown LLM pipeline type: {engine_type}")
