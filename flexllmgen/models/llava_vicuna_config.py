import numpy as np

class LlavaVicunaFlexConfig:
    # --- Text Config (Llama-based Main LLM) ---
    name: str = "llava-vicuna-7b"
    vocab_size: int = 32064
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    pad_token_id: int = 0
    eos_token_id: int = 2
    # hidden_act: str = "silu"
    # rms_norm_eps: float = 1e-05
    
    # --- Vision Config (CLIP-based) ---
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_depth: int = 24
    vision_patch_size: int = 14
    vision_in_chans: int = 3
    vision_num_positions: int = 577

    # --- Other Config ---
    dtype: type = np.float16
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
        Calculates the total parameter bytes (Vision Tower + Projector + Language Model).
        Based on the provided weights_llava.txt structure.
        """
        # --- 1. Vision Model Parameters (CLIP) ---
        # Embeddings
        # [cite_start]Patch Embed: Conv2d(3, 1024, k=14, s=14, bias=False) [cite: 599]
        vision_embed = (self.vision_in_chans * self.vision_hidden_size * \
                        self.vision_patch_size * self.vision_patch_size)
        # [cite_start]Position Embed: [577, 1024] [cite: 609]
        vision_pos_embed = self.vision_num_positions * self.vision_hidden_size
        # [cite_start]Class Embed: [1024] [cite: 608]
        vision_class_embed = self.vision_hidden_size
        
        # [cite_start]Pre-Layernorm: [1024] weight + bias [cite: 610]
        vision_pre_ln = 2 * self.vision_hidden_size

        # Vision Blocks (24 layers)
        # Structure: LayerNorm1 -> Attn(QKV+Out) -> LayerNorm2 -> MLP(FC1+FC2)
        # [cite_start]CLIP uses LayerNorm (affine=True, so weight+bias) and Linear layers with bias=True [cite: 600, 601, 602]
        vision_block_params = 0
        
        # LayerNorm 1 (weight + bias)
        vision_block_params += 2 * self.vision_hidden_size
        
        # Attn: k, v, q, out projections. [cite_start]All 1024->1024 with bias [cite: 601]
        # 4 projections * (weight + bias)
        vision_block_params += 4 * ((self.vision_hidden_size * self.vision_hidden_size) + self.vision_hidden_size)
        
        # LayerNorm 2 (weight + bias)
        vision_block_params += 2 * self.vision_hidden_size
        
        # MLP: fc1 (1024->4096), fc2 (4096->1024). [cite_start]Both with bias [cite: 602]
        # FC1
        vision_block_params += (self.vision_hidden_size * self.vision_intermediate_size) + self.vision_intermediate_size
        # FC2
        vision_block_params += (self.vision_intermediate_size * self.vision_hidden_size) + self.vision_hidden_size

        total_vision_blocks = vision_block_params * self.vision_depth
        
        # [cite_start]Post-Layernorm: [1024] weight + bias [cite: 660]
        vision_post_ln = 2 * self.vision_hidden_size

        total_vision_bytes = (vision_embed + vision_pos_embed + vision_class_embed + \
                              vision_pre_ln + total_vision_blocks + vision_post_ln) * self.bytes_per_param

        # --- 2. Multi-Modal Projector Parameters ---
        # Structure: Linear(1024->4096) -> GELU -> Linear(4096->4096)
        # [cite_start]weights_llava.txt confirms bias=True for these layers [cite: 603, 661, 663]
        projector_params = 0
        # Linear 1
        projector_params += (self.vision_hidden_size * self.hidden_size) + self.hidden_size
        # Linear 2
        projector_params += (self.hidden_size * self.hidden_size) + self.hidden_size
        
        # [cite_start]Image Newline parameter: [4096] [cite: 608]
        projector_params += self.hidden_size

        total_projector_bytes = projector_params * self.bytes_per_param

        # --- 3. Language Model Parameters (Llama) ---
        # Embeddings
        lm_embed = self.vocab_size * self.hidden_size
        
        # Decoder Layers (32 layers)
        # Llama usually has bias=False for Linear layers and RMSNorm (weight only)
        lm_layer_params = 0
        
        # [cite_start]Self Attention [cite: 604]
        # Q, K, V, O: All 4096->4096 (bias=False)
        lm_layer_params += 4 * (self.hidden_size * self.hidden_size)
        
        # [cite_start]MLP [cite: 605]
        # Gate, Up: 4096->11008 (bias=False)
        # Down: 11008->4096 (bias=False)
        lm_layer_params += 3 * (self.hidden_size * self.intermediate_size)
        
        # [cite_start]Norms (Input + Post-Attn) [cite: 605]
        # RMSNorm has weight but no bias
        lm_layer_params += 2 * self.hidden_size

        total_lm_layers = lm_layer_params * self.num_hidden_layers
        
        # [cite_start]Final Norm [cite: 837]
        final_norm = self.hidden_size
        
        # [cite_start]LM Head: 4096->32064 (bias=False) [cite: 838]
        lm_head = self.hidden_size * self.vocab_size

        total_lm_bytes = (lm_embed + total_lm_layers + final_norm + lm_head) * self.bytes_per_param

        return total_vision_bytes + total_projector_bytes + total_lm_bytes

    def cache_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        Calculates the KV Cache memory usage.
        Formula: 2 (rep. KV) * batch * seq_len * layers * kv_heads * head_dim * dtype_bytes
        Note: LLaVA-v1.5/1.6 (Vicuna/Llama based) typically uses Multi-Head Attention (MHA),
        so num_attention_heads (32) == num_attention_heads (32).
        """
        return 2 * batch_size * seq_len * self.num_hidden_layers * \
               self.num_attention_heads * self.head_dim * self.bytes_per_param

    def hidden_bytes(self, batch_size: int, seq_len: int) -> int:
        """
        Calculates the memory for hidden states (activations) output size.
        """
        return batch_size * seq_len * self.hidden_size * self.bytes_per_param
    

def get_llava_config(name, **kwargs):
    config = LlavaVicunaFlexConfig()
    return config


if __name__ == "__main__":
    config = get_llava_config("llava-vicuna-7b")
    weight_size = config.model_bytes()
    cache_size = config.cache_bytes(1, 2048)
    hidden_size = config.hidden_bytes(1, 2048)
    
    from flexllmgen.utils import GB
    print(f"model weight size: {weight_size/GB:.3f} GB, "
          f"kv cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")