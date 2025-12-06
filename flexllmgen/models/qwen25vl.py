from typing import Union, List, Optional, Tuple
import numpy as np
import torch
import os
from tqdm import tqdm

from flexllmgen.timer import timers
from flexllmgen.utils import (VisionTask, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from flexllmgen.policy import Policy
from flexllmgen.models.qwen25vl_config import Qwen25VLConfig, get_qwen25vl_config

from flexllmgen.pytorch_backend import TorchTensor

from flexllmgen.models.qwen_layers import (
    Qwen2_5_VLTextInputEmbed,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLAttention,
    Qwen2MLP,
    Qwen2_5_VLOutputHead,
)

from flexllmgen.models.base_model import BaseLM




from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLRotaryEmbedding
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

# modified from Qwen2_5_VLModel
def get_video_features(
    visual_encoder: Qwen2_5_VisionTransformerPretrainedModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None
):
    """
    Encodes videos into continuous embeddings that can be forwarded to the language model.
    """
    pixel_values_videos = pixel_values_videos.type(visual_encoder.dtype)
    video_embeds = visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
    split_sizes = (video_grid_thw.prod(-1) // visual_encoder.spatial_merge_size**2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds

# modified from Qwen2_5_VLModel
def get_placeholder_mask(
    text_embed_layer: torch.nn.Embedding,
    config: Qwen2_5_VLConfig,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: Optional[torch.FloatTensor] = None,
    video_features: Optional[torch.FloatTensor] = None,
):
    """
    Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
    equal to the length of multimodal features. If the lengths are different, an error is raised.
    """
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

# modified from Qwen2_5_VLModel
def get_rope_index(
    config: Qwen2_5_VLConfig,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask == 1
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i]]
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
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
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

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                ## normalize type, send to device.
                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                )

                time_tensor = expanded_range * second_per_grid_t * config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
            else:
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
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






class Qwen25VLLM(BaseLM):
    '''
    为简化实现，encoder部分并没有使用FlexGen的Layer组件构建，
    而是直接使用huggingface transformers的visual部分，完成
    视觉编码及两种模态token的merge。
    由于只关注模型decoding阶段指标（TPOT等），encoder部分的实现
    并不影响最终性能的测量。
    '''
    attention_layer_ids: List[int] = None
    rotary_emb: Qwen2_5_VLRotaryEmbedding = None
    rope_deltas: torch.Tensor = None # [batch size, 1]
    position_embeddings: Tuple[torch.Tensor] = None # (cos, sin), each [Sections(T/H/W), Batch, Seq_Len, Head_Dim]

    def get_model_config(self) -> Qwen25VLConfig:
        return get_qwen25vl_config(self.name)

    def init_model_layers(self) -> List:
        layers = []
        attention_layer_ids = []
        layers.append(Qwen2_5_VLTextInputEmbed(self.config, self.env, self.policy))
        cnt = 1
        for i in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                layers.append(Qwen2_5_VLAttention(self.config, self.env, self.policy, i))
                attention_layer_ids.append(cnt)
                layers.append(Qwen2MLP(self.config, self.env, self.policy, i))
                cnt += 2
            else:
                layers.append(Qwen2_5_VLDecoderLayer(self.config, self.env, self.policy, i))
                attention_layer_ids.append(cnt)
                cnt += 1
        layers.append(Qwen2_5_VLOutputHead(self.config, self.env, self.policy))
        self.attention_layer_ids = attention_layer_ids
        return layers

    def get_task(self, inputs, max_new_tokens, cut_gen_len, do_sample, temperature, stop) -> VisionTask:
        return VisionTask(
            input_ids=np.array(inputs.input_ids),
            prompt_len=inputs.input_ids.shape[1],
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,

            attention_mask=inputs.attention_mask,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            second_per_grid_ts=inputs.second_per_grid_ts,
        )

    def compute_layer(self, i, j, k):
        # 为attention层计算额外传入position_embeddings
        if j in self.attention_layer_ids:
            self.layers[j].forward(
                self.hidden[i][j][k], 
                self.cache_read_buf[j][k],
                self.weight_read_buf[j], 
                self.attention_mask[k],
                self.cache_write_buf[j][k], 
                i, k, self.position_embeddings
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
        if self.config.name == "qwen25vl-7b":
            model_path = '/data/lyc/models/Qwen2.5-VL-7B-Instruct'
        else:
            raise ValueError('Unimplemented.')
        qwen_config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        # load embedding layer and visual encoder
        text_embed_layer = torch.nn.Embedding(qwen_config.vocab_size, qwen_config.hidden_size, self.config.pad_token_id).to(device=device)
        visual_encoder = Qwen2_5_VisionTransformerPretrainedModel._from_config(qwen_config.vision_config).to(device=device)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=qwen_config).to(device=device)

        # get inputs
        input_ids = torch.tensor(self.task.input_ids, dtype=torch.int64).to(device=device)
        attention_mask = self.task.attention_mask.to(device=device)
        pixel_values_videos = self.task.pixel_values_videos.to(device=device)
        image_grid_thw = None
        video_grid_thw = self.task.video_grid_thw.to(device=device)
        second_per_grid_ts = self.task.second_per_grid_ts.to(device=device)

        # encode texts into embeddings
        inputs_embeds = text_embed_layer(input_ids) # [batch size, seq len, hidden dim]
        # encode videos into embeddings
        video_embeds = get_video_features(visual_encoder, pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.dtype)
        # merge video embeddings into text embeddings
        _, video_mask = get_placeholder_mask(text_embed_layer, qwen_config,
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds) # float32

        # set hidden
        hidden_states = inputs_embeds.to(dtype=torch.bfloat16)
        self.hidden[0][0][k].val = TorchTensor.create_from_torch(hidden_states, self.env.gpu)
        
        # set position embeddings
        position_ids, rope_deltas = get_rope_index(
            config=qwen_config, 
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask
        )
        position_ids = position_ids
        self.rope_deltas = rope_deltas
        self.position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # clear encoder weights
        del text_embed_layer
        del visual_encoder
    
    def set_position_embeddings(self, inputs_embeds, cur_pos_id):
        inputs_embeds = inputs_embeds.to(device=self.env.gpu.dev)
        
        # 获取当前生成的 token 数量，通常在解码阶段 seq_length 为 1
        batch_size, seq_length, _ = inputs_embeds.shape
        # 创建基础的相对位置索引 (0, 1, 2...)
        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        # 扩展维度以适配 Qwen2-VL 的 3D RoPE 结构
        # 形状变为 (3, batch_size, seq_length)，3 代表 (Time, Height, Width) 三个维度
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        # 计算绝对位置
        delta = (cur_pos_id + self.rope_deltas).to(inputs_embeds.device)
        # 广播 Delta 以匹配 batch 维度
        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
        # 最终位置 = 基础位置 + (全局计数 + 偏移量)
        position_ids = position_ids + delta.to(position_ids.device)
        # 使用 RoPE 计算位置嵌入
        self.position_embeddings = self.rotary_emb(inputs_embeds, position_ids)


    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                print(f'i={i}, j={j}')
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
                        # TODO: update position_ids
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    
                    # set position embeddings to be shared across all attention layers
                    # do after input embedding layer, before all attention layers
                    # Note: when (j == 0 and i == 0), set position embeddings is done in encoder()
                    if j == 0 and i != 0:
                        inputs_embeds = self.hidden[i][j][k].val.data
                        cur_pos_id = self.task.prompt_len + i
                        self.set_position_embeddings(inputs_embeds, cur_pos_id)

                    # print(f'Generation step {i}, layer {j}, batch {k} done.')
                    # import pdb; pdb.set_trace()
            
            timers("generate").stop()

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

