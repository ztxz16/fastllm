
class FastLLmModel:
    def __init__(self,
                 model_name,
                 ):
        self.model_name = model_name
        data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "fastllm",
                "permission": []
            }
        ]
        self.response = {
            "data": data,
            "object": "list"
        }