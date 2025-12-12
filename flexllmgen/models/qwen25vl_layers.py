'''
注：只实现了Qwen2.5-VL模型的decoder部分
'''


import os

from flexllmgen.policy import init_weight_list
from flexllmgen.models.opt_layers import InputEmbed, OutputEmbed, SelfAttention, MLP, TransformerLayer


class Qwen2_5_VLTextInputEmbed(InputEmbed):
    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "language_model.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token = weight_home.val[0] # weight_home.val赋值单变量需要解包
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst)))
    
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 2
        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[1]) = weight_read_buf.pop()
        else:
            (w_token, _) = weight_read_buf.val

        h = self.compute.qwen25vl_text_input_embed(h, w_token, self.config.pad_token_id, donate)
        hidden.val = h


class Qwen2_5_VLOutputHead(OutputEmbed):
    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "language_model.norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))
            
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.qwen25vl_output_embed(h, w_ln, w_token, donate,
            self.task.do_sample, self.task.temperature, self.config.rms_norm_eps)
        hidden.val = h



'''
DecoderLayer forward 伪代码：

residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(
    hidden_states, attention_mask, position_ids, position_embeddings, ...)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states

outputs = (hidden_states,)
'''

class Qwen2_5_VLAttention(SelfAttention):
    def init_weight(self, weight_home, path):
        h, n_head, n_kv_head, dtype = (self.config.input_dim, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.dtype)
        head_h = h // n_head * n_kv_head
        path = os.path.join(os.path.join(path, f"language_model.layers.{self.layer_id}."))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + "self_attn.q_proj.weight"),
            # b_q
            ((h,), dtype, path + "self_attn.q_proj.bias"),
            # w_k
            ((head_h, h), dtype, path + "self_attn.k_proj.weight"),
            # b_k
            ((head_h,), dtype, path + "self_attn.k_proj.bias"),
            # w_v
            ((head_h, h), dtype, path + "self_attn.v_proj.weight"),
            # b_v
            ((head_h,), dtype, path + "self_attn.v_proj.bias"),
            # w_out
            ((h, h), dtype, path + "self_attn.o_proj.weight"),
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), w_ln.smart_copy(dst2)))
            
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, position_embeddings):
        n_head = self.config.num_attention_heads
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 12
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (w_ln, donate[9])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (w_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.qwen25vl_gqa(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, w_ln, n_head, n_kv_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.config.rms_norm_eps, position_embeddings, self.config.rope_scaling_mrope_section)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[10]), (v_cache, donate[11]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.qwen25vl_gqa_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, w_ln, n_head, n_kv_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.config.rms_norm_eps, position_embeddings, self.config.rope_scaling_mrope_section)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class Qwen2_5_VLMLP(MLP):
    def init_weight(self, weight_home, path):
        h, mid_h, dtype = (self.config.input_dim, self.config.intermediate_size, self.config.dtype)
        path = os.path.join(os.path.join(path, f"language_model.layers.{self.layer_id}."))
        weight_specs = [
            # w_g
            ((mid_h, h), dtype, path + "mlp.gate_proj.weight"),
            # w_u
            ((mid_h, h), dtype, path + "mlp.up_proj.weight"),
            # w_d
            ((h, mid_h), dtype, path + "mlp.down_proj.weight"),
            # w_ln
            ((h,), dtype, path + "post_attention_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_g, w_u, w_d, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_g.smart_copy(dst1), w_u.smart_copy(dst1),
                w_d.smart_copy(dst1), w_ln.smart_copy(dst2)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_g, donate[1]), (w_u, donate[2]), (w_d, donate[3]), (w_ln, donate[4])) = weight_read_buf.pop()
        else:
            ((w_g, _), (w_u, _), (w_d, _), (w_ln, _)) = weight_read_buf.val

        assert self.config.hidden_act == "silu", "Only SiLU activation is supported in Qwen2.5-VL MLP"
        h = self.compute.qwen25vl_mlp(h, w_g, w_u, w_d, w_ln, donate, self.config.rms_norm_eps)
        hidden.val = h


class Qwen2_5_VLDecoderLayer(TransformerLayer):
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, position_embeddings):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k, position_embeddings)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)



