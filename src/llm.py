import torch
from transformers import pipeline

class LLMService:
    _instance = None

    @classmethod
    def get_instance(cls, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Determine dtype
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading local LLM {model_name} on device {self.device}...")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            torch_dtype=dtype,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print("Model loaded.")

    def generate_chat(self, messages: list, max_new_tokens: int = 200):
        # Apply chat template
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()
