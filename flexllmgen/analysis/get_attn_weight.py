import argparse
import os
import sys
from datetime import datetime
import torch

from flexllmgen.main import add_parser_arguments, print_args, run_flexllmgen_qwen
import flexllmgen.utils as utils



if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6

    # args.model_type = "qwen25vl-7b"
    args.model_type = "qwen3vl-8b"

    args.attn_impl = "eager"  # only for no sparse

    video_id = '28s'
    video_path = "./test/video/28s.mp4"
    question = "Please describe this video in detail."

    utils.total_attn_weight.setup(args.model_type, video_id)

    # 项目入口
    if args.model_type == 'qwen25vl-7b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    elif args.model_type == 'qwen3vl-8b':
        run_flexllmgen_qwen(args, video_path=video_path, question=question)
    else:
        raise ValueError(f"Unsupported model: {args.model_type}")


    # 保存attn_weight
    utils.total_attn_weight.save()
    print(f'max step: {utils.total_attn_weight.max_step}')