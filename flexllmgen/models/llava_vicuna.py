
'''
config.json中有以下参数：
"max_position_embeddings": 4096
"rope_scaling" : {
  "factor": 2.5,
  "type": "linear"
}
说明模型可以处理的有效长度最多是 4096 * 2.5 = 10240 个 token。

而Qwen2.5-VL模型的max_position_embeddings=128000，
Qwen3-VL模型的max_position_embeddings=262144，远远大于llava。

因此考虑不使用llava模型。
'''

