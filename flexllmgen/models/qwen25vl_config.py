import dataclasses
import torch
import numpy as np

@dataclasses.dataclass(frozen=True)
class Qwen25VLConfig:
    # --- Language Model Config ---
    name: str = "qwen25vl-7b"
    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4  # GQA: KV heads count
    intermediate_size: int = 18944
    pad_token_id: int = 151643
    eos_token_id: tuple[int] = (151645, 151643)
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-06
    rope_scaling_mrope_section: list[int] = dataclasses.field(default_factory=lambda: [16, 24, 24])
    
    # --- Vision Config ---
    vision_hidden_size: int = 1280
    vision_intermediate_size: int = 3420
    vision_depth: int = 32
    vision_patch_size: int = 14
    vision_in_chans: int = 3
    vision_temporal_patch_size: int = 2
    
    dtype: type = np.uint16 # <==> torch.bfloat16
    bytes_per_param: int = 2

    @property
    def n_head(self) -> int:
        return self.num_attention_heads

    @property
    def input_dim(self) -> int:
        return self.hidden_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def model_bytes(self) -> int:
        """
        Calculates the total parameter bytes (Language Model + Vision Encoder).
        Based on the provided print(model) structure.
        """
        # --- 1. Vision Model Parameters ---
        # Patch Embed: Conv3d(3, 1280, k=(2,14,14))
        # Weight: out * in * k_t * k_h * k_w
        vision_patch_embed = self.vision_hidden_size * self.vision_in_chans * \
                             self.vision_temporal_patch_size * \
                             self.vision_patch_size * self.vision_patch_size

        # Vision Blocks (32 layers)
        # Structure: Norm -> Attn(QKV+Proj) -> Norm -> MLP(Gate+Up+Down)
        # Note: print(model) shows QKV is one Linear layer (1280 -> 3840) with Bias=True
        vision_block_params = 0
        
        # Attn QKV (in: 1280, out: 3840, bias=True)
        vision_block_params += (self.vision_hidden_size * (self.vision_hidden_size * 3)) + (self.vision_hidden_size * 3)
        # Attn Proj (in: 1280, out: 1280, bias=True)
        vision_block_params += (self.vision_hidden_size * self.vision_hidden_size) + self.vision_hidden_size
        
        # MLP (Gate, Up, Down). All have bias=True in Vision transformer usually, 
        # print(model) confirms bias=True for all linear layers in visual block.
        # Gate: in->inter, Up: in->inter, Down: inter->in
        vision_block_params += 2 * (self.vision_hidden_size * self.vision_intermediate_size + self.vision_intermediate_size) # Gate + Up
        vision_block_params += (self.vision_intermediate_size * self.vision_hidden_size + self.vision_hidden_size) # Down

        # Norms (2 per block, RMSNorm usually has weight but no bias parameter in implementation, 
        # but print shows (1280,), counting scale parameter)
        vision_block_params += 2 * self.vision_hidden_size

        total_vision_blocks = vision_block_params * self.vision_depth
        
        # Vision Rot/Norm parameters (minor, but adding generic Norm)
        total_vision_misc = self.vision_hidden_size # extra norm

        # --- 2. Patch Merger Parameters ---
        # Structure: Norm -> Linear(5120->5120) -> GELU -> Linear(5120->3584)
        # Input 5120 comes from spatial_merge_size=2 (1280 * 2*2 = 5120)
        merger_in_dim = self.vision_hidden_size * 4 
        merger_params = 0
        merger_params += self.vision_hidden_size # ln_q
        merger_params += (merger_in_dim * merger_in_dim) + merger_in_dim # Linear 1 + bias
        merger_params += (merger_in_dim * self.hidden_size) + self.hidden_size # Linear 2 + bias

        total_vision_bytes = (vision_patch_embed + total_vision_blocks + total_vision_misc + merger_params) * self.bytes_per_param

        # --- 3. Language Model Parameters ---
        # Embeddings (No bias)
        lm_embed = self.vocab_size * self.hidden_size
        
        # Decoder Layers (28 layers)
        lm_layer_params = 0
        
        # Self Attention
        # Q: hidden -> hidden (bias=True)
        lm_layer_params += (self.hidden_size * self.hidden_size) + self.hidden_size
        # K: hidden -> kv_heads * head_dim (bias=True)
        kv_dim = self.num_key_value_heads * self.head_dim
        lm_layer_params += (self.hidden_size * kv_dim) + kv_dim
        # V: hidden -> kv_heads * head_dim (bias=True)
        lm_layer_params += (self.hidden_size * kv_dim) + kv_dim
        # O: hidden -> hidden (bias=False)
        lm_layer_params += (self.hidden_size * self.hidden_size)
        
        # MLP (Gate, Up, Down) - All bias=False
        # Gate & Up: hidden -> intermediate
        lm_layer_params += 2 * (self.hidden_size * self.intermediate_size)
        # Down: intermediate -> hidden
        lm_layer_params += (self.intermediate_size * self.hidden_size)
        
        # Norms (Input + Post-Attn)
        lm_layer_params += 2 * self.hidden_size

        total_lm_layers = lm_layer_params * self.num_hidden_layers
        
        # Final Norm & Head
        final_norm = self.hidden_size
        # LM Head: hidden -> vocab (bias=False)
        lm_head = self.hidden_size * self.vocab_size

        total_lm_bytes = (lm_embed + total_lm_layers + final_norm + lm_head) * self.bytes_per_param

        return total_vision_bytes + total_lm_bytes

    def cache_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        Calculates the KV Cache memory usage.
        Formula: 2 (rep. KV) * batch * seq_len * layers * kv_heads * head_dim * dtype_bytes
        Note: Qwen2.5 uses GQA, so we use num_key_value_heads (4), not num_attention_heads (28).
        """
        return 2 * batch_size * seq_len * self.num_hidden_layers * \
               self.num_key_value_heads * self.head_dim * self.bytes_per_param

    def hidden_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        Calculates the memory for hidden states (activations) output size.
        Note: Only achieve maximum in prefill stage.
        """
        return batch_size * seq_len * self.hidden_size * self.bytes_per_param
    

def get_qwen25vl_config(name, **kwargs):
    # TODO: 根据模型大小设置不同参数
    config = Qwen25VLConfig()
    return dataclasses.replace(config, **kwargs)


if __name__ == "__main__":
    config = get_qwen25vl_config("qwen25vl-7b")
    weight_size = config.model_bytes()
    cache_size = config.cache_bytes(1, 2048)
    hidden_size = config.hidden_bytes(1, 2048)
    from flexllmgen.utils import GB
    print(f"model weight size: {weight_size/GB:.3f} GB, "
          f"kv cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")