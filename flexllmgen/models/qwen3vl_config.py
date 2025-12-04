import dataclasses

@dataclasses.dataclass(frozen=True)
class Qwen3VLConfig:
    # --- Text Config (Main LLM) ---
    name: str = "qwen3vl"
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: KV heads count
    intermediate_size: int = 12288
    
    # --- Vision Config ---
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_depth: int = 27
    vision_patch_size: int = 16
    vision_in_chans: int = 3
    vision_temporal_patch_size: int = 2
    vision_num_pos_embeddings: int = 2304
    vision_output_hidden_size: int = 4096 # Dimension after merger
    
    # DeepStack specifics
    deepstack_layers: int = 3 # Length of deepstack_merger_list
    
    bytes_per_param: int = 2 # bfloat16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def model_bytes(self) -> int:
        """
        Calculates total parameter bytes.
        Adapts to Qwen3-VL specifics: Vision MLP change, DeepStack, QK-Norms.
        """
        # ==========================================
        # 1. Vision Model (Qwen3VLVisionModel)
        # ==========================================
        
        # --- A. Embeddings ---
        # Patch Embed: Conv3d(3, 1152, k=(2,16,16))
        # Assuming bias=True (standard Conv3d default, though not explicitly verified as False)
        vision_patch_w = self.vision_hidden_size * self.vision_in_chans * \
                         self.vision_temporal_patch_size * \
                         self.vision_patch_size * self.vision_patch_size
        vision_patch_b = self.vision_hidden_size
        
        # Pos Embed: Embedding(2304, 1152) - Explicit in Qwen3
        vision_pos_embed = self.vision_num_pos_embeddings * self.vision_hidden_size
        
        vision_embed_params = vision_patch_w + vision_patch_b + vision_pos_embed

        # --- B. Vision Blocks (27 layers) ---
        # Structure: Norm1 -> Attn -> Norm2 -> MLP(FC1-Act-FC2)
        # Note: Vision Linear layers explicitly show bias=True
        v_block_params = 0
        
        # Norms (LayerNorm has weight & bias)
        v_block_params += 2 * (2 * self.vision_hidden_size) # Norm1 + Norm2
        
        # Attention
        # QKV: 1152 -> 3456 (bias=True)
        v_block_params += (self.vision_hidden_size * (self.vision_hidden_size * 3)) + (self.vision_hidden_size * 3)
        # Proj: 1152 -> 1152 (bias=True)
        v_block_params += (self.vision_hidden_size * self.vision_hidden_size) + self.vision_hidden_size
        
        # MLP (CHANGED in Qwen3: Standard MLP, not SwiGLU)
        # FC1: 1152 -> 4304 (bias=True)
        v_block_params += (self.vision_hidden_size * self.vision_intermediate_size) + self.vision_intermediate_size
        # FC2: 4304 -> 1152 (bias=True)
        v_block_params += (self.vision_intermediate_size * self.vision_hidden_size) + self.vision_hidden_size

        total_vision_blocks = v_block_params * self.vision_depth
        
        # Rotary Embedding (fixed, no params) & Misc
        
        # --- C. Main Merger ---
        # Input to merger logic is usually spatial_merge_size^2 * hidden = 2*2*1152 = 4608
        merger_in_dim = 4608
        merger_params = 0
        
        # Norm: LayerNorm((1152,)) - Note: print shows 1152, likely applied before concat or per-channel
        merger_params += 2 * self.vision_hidden_size
        
        # Linear FC1: 4608 -> 4608 (bias=True)
        merger_params += (merger_in_dim * merger_in_dim) + merger_in_dim
        # Linear FC2: 4608 -> 4096 (bias=True)
        merger_params += (merger_in_dim * self.vision_output_hidden_size) + self.vision_output_hidden_size

        # --- D. DeepStack Mergers (3 layers) ---
        # These are extra merger layers used for intermediate visual features
        # Structure is similar to Main Merger but Norm is explicitly (4608,) in print
        deepstack_single_params = 0
        deepstack_single_params += 2 * merger_in_dim # Norm (weight+bias)
        deepstack_single_params += (merger_in_dim * merger_in_dim) + merger_in_dim # FC1
        deepstack_single_params += (merger_in_dim * self.vision_output_hidden_size) + self.vision_output_hidden_size # FC2
        
        total_deepstack = deepstack_single_params * self.deepstack_layers

        total_vision_bytes = (vision_embed_params + total_vision_blocks + merger_params + total_deepstack) * self.bytes_per_param

        # ==========================================
        # 2. Text Model (Qwen3VLTextModel)
        # ==========================================
        
        # Embeddings (vocab -> hidden)
        lm_embed = self.vocab_size * self.hidden_size
        
        # Decoder Layers (36 layers)
        lm_layer_params = 0
        
        # Self Attention (bias=False)
        # Q: hidden -> hidden
        lm_layer_params += (self.hidden_size * self.hidden_size)
        # K, V: hidden -> kv_heads * head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        lm_layer_params += 2 * (self.hidden_size * kv_dim)
        # O: hidden -> hidden
        lm_layer_params += (self.hidden_size * self.hidden_size)
        
        # **NEW in Qwen3: QK Norms**
        # Qwen3VLTextRMSNorm((128,)) -> head_dim size.
        # Assuming parameters are shared across heads (broadcast) or defined as (head_dim,).
        # Print output `(128,)` strongly implies just 128 params per norm layer.
        lm_layer_params += self.head_dim + self.head_dim # q_norm + k_norm
        
        # MLP (SwiGLU, bias=False)
        # Gate + Up: hidden -> intermediate
        lm_layer_params += 2 * (self.hidden_size * self.intermediate_size)
        # Down: intermediate -> hidden
        lm_layer_params += (self.intermediate_size * self.hidden_size)
        
        # Layer Norms (RMSNorm, scale only)
        lm_layer_params += 2 * self.hidden_size # input_layernorm + post_attention

        total_lm_layers = lm_layer_params * self.num_hidden_layers
        
        # Final Norm
        final_norm = self.hidden_size
        # LM Head (bias=False)
        lm_head = self.hidden_size * self.vocab_size

        total_lm_bytes = (lm_embed + total_lm_layers + final_norm + lm_head) * self.bytes_per_param

        return total_vision_bytes + total_lm_bytes

    def cache_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        KV Cache Memory.
        Formula: 2 * batch * seq * layers * kv_heads * head_dim * dtype_bytes
        """
        return 2 * batch_size * seq_len * self.num_hidden_layers * \
               self.num_key_value_heads * self.head_dim * self.bytes_per_param

    def hidden_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        Activation Memory (Output Hidden States).
        """
        return batch_size * seq_len * self.hidden_size * self.bytes_per_param
    

def get_qwen3vl_config(name, **kwargs):
    config = Qwen3VLConfig()
    return dataclasses.replace(config, **kwargs)


if __name__ == "__main__":
    config = get_qwen3vl_config("qwen3vl-8b")
    weight_size = config.model_bytes()
    cache_size = config.cache_bytes(1, 2048)
    hidden_size = config.hidden_bytes(1, 2048)
    from flexllmgen.utils import GB
    print(f"model weight size: {weight_size/GB:.3f} GB, "
          f"kv cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")