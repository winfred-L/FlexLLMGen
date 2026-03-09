
class Qwen3VLConfig:
    model_path: str = None
    device: str = 'cuda:0'

    def __init__(self, model_type: str = 'Qwen3-VL-8B-Instruct'):
        if model_type == 'Qwen3-VL-8B-Instruct':
            self.model_path = '/data/lyc/models/Qwen3-VL-8B-Instruct'
    