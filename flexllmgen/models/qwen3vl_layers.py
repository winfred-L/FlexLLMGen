import os

from flexllmgen.policy import init_weight_list
from flexllmgen.models.qwen25vl_layers import (
    Qwen2_5_VLTextInputEmbed,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLAttention,
    Qwen2_5_VLMLP,
    Qwen2_5_VLOutputHead,
)


class Qwen3VLTextInputEmbed(Qwen2_5_VLTextInputEmbed):
    pass

class Qwen3VLOutputHead(Qwen2_5_VLOutputHead):
    pass


class Qwen3VLTextAttention(Qwen2_5_VLAttention):
    def init_weight(self, weight_home, path):
        h, n_head, n_kv_head, dtype = (self.config.input_dim, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.dtype)
        head_h = h // n_head * n_kv_head
        head_dim = self.config.head_dim
        path = os.path.join(os.path.join(path, f"language_model.layers.{self.layer_id}."))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((head_h, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((head_h, h), dtype, path + "self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + "self_attn.o_proj.weight"),
            # q_ln
            ((head_dim,), dtype, path + "self_attn.q_norm.weight"),
            # k_ln
            ((head_dim,), dtype, path + "self_attn.k_norm.weight"),
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, w_k, w_v, w_out, q_ln, k_ln, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), w_k.smart_copy(dst1),
                w_v.smart_copy(dst1), w_out.smart_copy(dst1),
                q_ln.smart_copy(dst2), k_ln.smart_copy(dst2),
                w_ln.smart_copy(dst2)))
            
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, position_embeddings):
        n_head = self.config.num_attention_heads
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 11
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), (w_out, donate[5]),
             (q_ln, donate[6]), (k_ln, donate[7]), (w_ln, donate[8])) = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _), (w_v, _), (w_out, _),
             (q_ln, _), (k_ln, _), (w_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.qwen3vl_gqa(h, mask, w_q,
                w_k, w_v, w_out, q_ln, k_ln, w_ln, n_head, n_kv_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.config.rms_norm_eps, position_embeddings)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            if self.policy.do_sparse: # sparse attention
                mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
                (k_cache, donate[9]), (v_cache, donate[10]) = cache_read_buf.pop()
                h, new_k_cache, new_v_cache = self.compute.qwen3vl_gqa_gen_sparse(h, mask, w_q,
                    w_k, w_v, w_out, q_ln, k_ln, w_ln, n_head, n_kv_head,
                    k_cache, v_cache, donate, self.policy.attn_sparsity,
                    self.policy.compress_cache, self.policy.comp_cache_config,
                    self.config.rms_norm_eps, position_embeddings)
                cache_write_buf.store((new_k_cache, new_v_cache))
            else: # dense attention
                mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
                (k_cache, donate[9]), (v_cache, donate[10]) = cache_read_buf.pop()
                h, new_k_cache, new_v_cache = self.compute.qwen3vl_gqa_gen(h, mask, w_q,
                    w_k, w_v, w_out, q_ln, k_ln, w_ln, n_head, n_kv_head,
                    k_cache, v_cache, donate, self.policy.attn_sparsity,
                    self.policy.compress_cache, self.policy.comp_cache_config,
                    self.config.rms_norm_eps, position_embeddings)
                cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class Qwen3VLTextMLP(Qwen2_5_VLMLP):
    def forward(self, hidden, cache_read_buf, weight_read_buf, 
                attention_mask, cache_write_buf, i, k,
                visual_pos_masks, deepstack_video_embeds):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_g, donate[1]), (w_u, donate[2]), (w_d, donate[3]), (w_ln, donate[4])) = weight_read_buf.pop()
        else:
            ((w_g, _), (w_u, _), (w_d, _), (w_ln, _)) = weight_read_buf.val

        assert self.config.hidden_act == "silu", "Only SiLU activation is supported in Qwen3-VL MLP"
        h = self.compute.qwen3vl_mlp(h, w_g, w_u, w_d, w_ln, donate, self.config.rms_norm_eps, visual_pos_masks, deepstack_video_embeds)
        hidden.val = h


class Qwen3VLTextDecoderLayer(Qwen2_5_VLDecoderLayer):
    def forward(self, hidden, cache_read_buf, weight_read_buf,
        attention_mask, cache_write_buf, i, k,
        position_embeddings, visual_pos_masks, deepstack_video_embeds,
    ):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1,
                               attention_mask, cache_write_buf, i, k,
                               position_embeddings)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k, visual_pos_masks, deepstack_video_embeds)

