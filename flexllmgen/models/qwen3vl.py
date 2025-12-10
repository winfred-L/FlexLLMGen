from typing import Union, List, Optional, Tuple
import numpy as np
import torch
import os
import gc

from flexllmgen.timer import timers
from flexllmgen.utils import VisionTask
from flexllmgen.pytorch_backend import TorchTensor
from flexllmgen.models.base_model import BaseFlexLM
from flexllmgen.models.qwen3vl_config import Qwen3VLFlexConfig, get_qwen3vl_config
from flexllmgen.models.qwen3vl_layers import (
    Qwen3VLTextInputEmbed,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextDecoderLayer,
    Qwen3VLOutputHead,
)

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLTextRotaryEmbedding


# modified from Qwen3VLModel
def get_video_features(
    visual_encoder: Qwen3VLVisionModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None
):
    pixel_values_videos = pixel_values_videos.to(visual_encoder.dtype)
    video_embeds, deepstack_video_embeds = visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
    split_sizes = (video_grid_thw.prod(-1) // visual_encoder.spatial_merge_size**2).tolist()
    image_embeds = torch.split(video_embeds, split_sizes)
    return image_embeds, deepstack_video_embeds

# modified from Qwen3VLModel
def get_placeholder_mask(
    text_embed_layer: torch.nn.Embedding,
    config: Qwen3VLConfig,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: Optional[torch.FloatTensor] = None,
    video_features: Optional[torch.FloatTensor] = None,
):
    if input_ids is None:
        special_image_mask = inputs_embeds == text_embed_layer(
            torch.tensor(config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
        special_video_mask = inputs_embeds == text_embed_layer(
            torch.tensor(config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_video_mask = special_video_mask.all(-1)
    else:
        special_image_mask = input_ids == config.image_token_id
        special_video_mask = input_ids == config.video_token_id

    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    n_video_tokens = special_video_mask.sum()
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        raise ValueError(
            f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        )

    return special_image_mask, special_video_mask

# modified from Qwen3VLModel
def get_rope_index(
    config: Qwen3VLConfig,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas





class Qwen3VLFlexLM(BaseFlexLM):
    '''
    Qwen3VL相较于Qwen2.5VL的改变如下：
    1. 视觉位置编码get_rope_index()实现不同，采用时间戳而非绝对位置编码
    2. 添加了deepstack_merger，会将视觉特征与decoder前几层的hidden state融合
    3. attention层添加了QK norm，作用在W_q与W_k之后，rotary_pos_emb之前
    '''

    attention_layer_ids: List[int] = None
    mlp_layer_ids: List[int] = None

    rotary_emb: Qwen3VLTextRotaryEmbedding = None
    rope_deltas: torch.Tensor = None # [batch size, 1]
    position_embeddings: Tuple[torch.Tensor] = None # (cos, sin), each [Sections(T/H/W), Batch, Seq_Len, Head_Dim]

    # args for deepstack merger
    visual_pos_masks: torch.Tensor = None # shape: [batch_size, visual_seqlen]
    deepstack_video_embeds: torch.Tensor = None # shape: [num_layers, visual_seqlen, embed_dim]


    def get_model_config(self) -> Qwen3VLFlexConfig:
        return get_qwen3vl_config(self.name)
    

    def init_model_layers(self) -> List:
        layers = []
        attention_layer_ids = []
        mlp_layer_ids = []
        layers.append(Qwen3VLTextInputEmbed(self.config, self.env, self.policy))
        cnt = 1
        for i in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                layers.append(Qwen3VLTextAttention(self.config, self.env, self.policy, i))
                attention_layer_ids.append(cnt)
                layers.append(Qwen3VLTextMLP(self.config, self.env, self.policy, i))
                mlp_layer_ids.append(cnt + 1)
                cnt += 2
            else:
                layers.append(Qwen3VLTextDecoderLayer(self.config, self.env, self.policy, i))
                attention_layer_ids.append(cnt)
                mlp_layer_ids.append(cnt)
                cnt += 1
        layers.append(Qwen3VLOutputHead(self.config, self.env, self.policy))
        self.attention_layer_ids = attention_layer_ids
        self.mlp_layer_ids = mlp_layer_ids
        return layers
    

    def get_task(self, inputs, max_new_tokens, cut_gen_len, do_sample, temperature, stop) -> VisionTask:
        return VisionTask(
            input_ids=np.array(inputs.input_ids),
            prompt_len=inputs.input_ids.shape[1],
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=self.config.eos_token_id if stop is None else stop,

            attention_mask=inputs.attention_mask,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            second_per_grid_ts=None, # qwen3vl没有这个参数
        )
    

    def compute_layer(self, i, j, k):
        '''
        为attention层计算额外传入position_embeddings
        为mlp层计算额外传入visual_pos_masks, deepstack_video_embeds
        '''
        if j in self.attention_layer_ids and j in self.mlp_layer_ids:
            # no sep layer case
            self.layers[j].forward(
                self.hidden[i][j][k], 
                self.cache_read_buf[j][k],
                self.weight_read_buf[j], 
                self.attention_mask[k],
                self.cache_write_buf[j][k], 
                i, k,
                self.position_embeddings,
                self.visual_pos_masks,
                self.deepstack_video_embeds,
            )
        elif j in self.attention_layer_ids:
            self.layers[j].forward(
                self.hidden[i][j][k], 
                self.cache_read_buf[j][k],
                self.weight_read_buf[j], 
                self.attention_mask[k],
                self.cache_write_buf[j][k], 
                i, k,
                self.position_embeddings,
            )
        elif j in self.mlp_layer_ids:
            # add visual features to the hidden states of first several decoder layers (only prefill)
            mlp_layer_index = self.mlp_layer_ids.index(j)
            if i == 0 and mlp_layer_index in range(len(self.deepstack_video_embeds)):
                self.layers[j].forward(
                    self.hidden[i][j][k], 
                    self.cache_read_buf[j][k],
                    self.weight_read_buf[j], 
                    self.attention_mask[k],
                    self.cache_write_buf[j][k], 
                    i, k,
                    self.visual_pos_masks,
                    self.deepstack_video_embeds[mlp_layer_index],
                )
            else:
                self.layers[j].forward(
                    self.hidden[i][j][k], 
                    self.cache_read_buf[j][k],
                    self.weight_read_buf[j], 
                    self.attention_mask[k],
                    self.cache_write_buf[j][k], 
                    i, k,
                    None,
                    None,
                )
        else:
            self.layers[j].forward(
                self.hidden[i][j][k], 
                self.cache_read_buf[j][k],
                self.weight_read_buf[j], 
                self.attention_mask[k],
                self.cache_write_buf[j][k], 
                i, k
            )


    def encoder(self, k):
        '''
        huggingface transformers中视觉编码器的原始实现，
        包含视频编码与模态拼接。
        '''
        assert k == 0, "Only support single GPU batch for encoder."

        device = self.env.gpu.dev

        # load model config
        if self.config.name == "qwen3vl-8b":
            model_path = '/data/lyc/models/Qwen3-VL-8B-Instruct'
            load_weights_path = '/data/lyc/models/qwen3vl-8b-np'
        else:
            raise NotImplementedError(f"Model {self.config.name} not supported yet.")
        qwen_config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        # load embedding layer and visual encoder
        text_embed_layer = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)
        visual_encoder = Qwen3VLVisionModel._from_config(qwen_config.vision_config)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=qwen_config.text_config)

        # load weights
        text_embed_layer.load_state_dict(torch.load(
            os.path.join(load_weights_path, 'text_embed_layer.bin'), 
            map_location='cpu'
        ))
        visual_encoder.load_state_dict(torch.load(
            os.path.join(load_weights_path, 'visual_encoder.bin'), 
            map_location='cpu'
        ))

        text_embed_layer = text_embed_layer.to(device=device).eval()
        visual_encoder = visual_encoder.to(device=device).eval()
        self.rotary_emb = self.rotary_emb.to(device=device).eval()

        # get inputs
        input_ids = torch.tensor(self.task.input_ids, dtype=torch.int64).to(device=device)
        attention_mask = self.task.attention_mask.to(device=device)
        pixel_values_videos = self.task.pixel_values_videos.to(device=device)
        image_grid_thw = None
        video_grid_thw = self.task.video_grid_thw.to(device=device)

        # encode texts into embeddings
        inputs_embeds = text_embed_layer(input_ids) # [batch size, seq len, hidden dim]
        # encode videos into embeddings
        video_embeds, deepstack_video_embeds = get_video_features(visual_encoder, pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = get_placeholder_mask(
            text_embed_layer, qwen_config,
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # set deepstack merger args
        self.visual_pos_masks = video_mask[..., 0]
        self.deepstack_video_embeds = deepstack_video_embeds
        
        # set hidden
        hidden_states = inputs_embeds.to(dtype=torch.bfloat16)
        self.hidden[0][0][k].val = TorchTensor.create_from_torch(hidden_states, self.env.gpu)

        # get attention_mask_tensor
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()
        
        # set position embeddings
        position_ids, rope_deltas = get_rope_index(
            qwen_config,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        self.rope_deltas = rope_deltas
        self.position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # clear encoder weights
        del text_embed_layer
        del visual_encoder
        torch.cuda.empty_cache()
        gc.collect()


    def set_position_embeddings(self, inputs_embeds, cur_pos_id):
        inputs_embeds = inputs_embeds.to(device=self.env.gpu.dev)

        batch_size, seq_length, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        delta = (cur_pos_id + self.rope_deltas).to(inputs_embeds.device)
        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        position_ids = position_ids.add(delta)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        self.position_embeddings = self.rotary_emb(inputs_embeds, position_ids)


    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                # print(f'i={i}, j={j}')
                for k in range(self.num_gpu_batches):
                    if j == 0 and i == 0:
                        pass # skip loading the first InputEmbed layer
                    else:
                        self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    if j == 0 and i == 0: # replace the first InputEmbed layer with visual encoder
                        self.encoder(k)
                    else:
                        self.compute_layer(i, j, k)
                    
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    
                    # set position embeddings to be shared across all attention layers
                    # do after input embedding layer, before all attention layers
                    # Note: when (j == 0 and i == 0), set position embeddings is done in encoder()
                    if j == 0 and i != 0:
                        inputs_embeds = self.hidden[i][j][k].val.data
                        cur_pos_id = self.task.prompt_len + i - 1
                        self.set_position_embeddings(inputs_embeds, cur_pos_id)
            
            timers("generate").stop()

            # print(f'i={i}, output_ids={self.output_ids[0, self.task.prompt_len + i]}')
            # import pdb; pdb.set_trace()

            # stop when all batches are stopped
            if np.all(self.stopped):
                break

    def generation_loop_debug_normal(self):
        raise ValueError('Unimplemented')

    def generation_loop_overlap_single_batch(self):
        raise ValueError('Unimplemented')

    def generation_loop_overlap_multi_batch(self):
        raise ValueError('Unimplemented')

    def generation_loop_debug_single_batch(self):
        raise ValueError('Unimplemented')

    def generation_loop_debug_multi_batch(self):
        raise ValueError('Unimplemented')