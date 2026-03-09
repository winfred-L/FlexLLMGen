


class Qwen3VLModel:

    def __init__(self, config):
        self.config = config
        self.model = self.load_model()


    def load_model(self):
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.config.device).eval()
        return model


    def generate(self, **inputs):
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        return generated_ids