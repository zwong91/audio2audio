from typing import List, Tuple, Dict

class LLMInterface:
    async def generate(self, query: str, max_tokens: int = 64) -> Tuple[str, List[Dict[str, str]]]:
        """
        根据对话历史生成回复
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
