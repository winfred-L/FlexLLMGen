'''
Engine 负责控制计算流程
 负责内存管理
Model 负责加载模型和提供计算接口
'''


from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from flexcachegen.config import Qwen3VLConfig
from flexcachegen.model import Qwen3VLModel



class Qwen3VLEngine:

    def __init__(self, model_type: str):
        self.config = Qwen3VLConfig(model_type)
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.model = Qwen3VLModel(self.config)

        self.device = self.config.device

    def process_input(self, video_path: str, question: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        # restrict the resolution of individual frames in the video
                        # "min_pixels": 4 * 32 * 32,
                        # "max_pixels": 640 * 32 * 32,
                        # limit the total number of tokens in the video
                        "total_pixels": 32 * 1024 * 32 * 32,
                        # accept either `fps` or `nframes`
                        # "fps": 2.0,
                        "nframes": 32, #2048,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        # each video returns as (video_tensor, video_metadata)
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False, # avoid duplicate resizing
            **video_kwargs
        )
        return inputs

    def generate_single(self, video_path: str, question: str) -> str:
        output_ids = []

        # 1. process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # 2. encoding step
        hidden_states = self.model.encoding(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"]
        )

        # 3. prefill stage
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(True, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
            hidden_states = self.model.merge_visual_features(hidden_states, layer_idx)

        token_id, logits = self.model.output_head(hidden_states)
        output_ids.append(token_id)

        # 4. decoding stage
        while not self.is_finished(output_ids):
            hidden_states = self.model.text_embed(token_id)
            cur_pos_id = prompt_len + len(output_ids) - 1
            self.model.set_rotary_pos_emb(hidden_states, cur_pos_id)

            for layer_idx in range(self.num_hidden_layers):
                hidden_states = self.model.attention(False, hidden_states, layer_idx)
                hidden_states = self.model.mlp(hidden_states, layer_idx)

            token_id, logits = self.model.output_head(hidden_states)
            output_ids.append(token_id)

            print(f"step {len(output_ids)}: token_id={token_id}")

        # 5. decode output
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text