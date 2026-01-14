import requests
import time
import logging
import ast

# 模拟源代码的全局配置
# API_URL = "https://yunwu.ai/v1/chat/completions"
API_URL = "https://yunwu.zeabur.app/v1/chat/completions"
GPT_EVAL_MODEL_NAME = "gpt-5-mini-2025-08-07"
headers = {"Authorization": "Bearer sk-yb4fxa2WVsZunr0G2OHz56QPjTHxgSnYBJD6GZE6EUYtK1ZN", "Content-Type": "application/json"}
NUM_SECONDS_TO_SLEEP = 1

def get_eval_generic_exact(question, answer, pred, task, max_tokens=64, retries=2):
    # --- 1. 完全还原源代码中的 Prompt 构造逻辑 ---
    if task == "correctness":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                "- The predicted answer must be factually accurate and align with the video content.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the factual accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    # ... 其他 task 逻辑一致 ...
    else:
        messages = [{"role": "user", "content": "dummy"}]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    # --- 2. 还原源代码中的循环和异常处理坑点 ---
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]

        except requests.exceptions.RequestException as e:
            # 源代码里：此处记录了日志，但 e 的生命周期仅限此 block
            print(f"DEBUG: Attempt {attempt + 1} 捕获到网络异常")
        
        except Exception as e:
            print(f"DEBUG: Attempt {attempt + 1} 捕获到通用异常")

        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:
            # --- 3. 报错触发位置 ---
            # 源代码逻辑：在 for 循环的 else 分支访问 e
            print("DEBUG: 正在执行最后一次失败后的逻辑...")
            try:
                # 这一行会引发 UnboundLocalError，因为 e 已被销毁
                print(f"All {retries} attempts failed. Last error message: {e}")
            except UnboundLocalError as err:
                print(f"\n【复现成功】抛出错误: {err}")
                return "触发了变量作用域错误", ""

    return "", ""

def main():
    # 模拟输入数据
    question = "What is the person doing?"
    answer = "The person is playing basketball."
    pred = "The person is playing football."
    
    print("开始测试 'correctness' 任务评估...")
    result, model = get_eval_generic_exact(question, answer, pred, "correctness")
    print(f"\n最终返回结果: {result}")

if __name__ == "__main__":
    main()