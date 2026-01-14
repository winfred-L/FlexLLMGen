from openai import OpenAI
import json
import os
import ast

EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5-mini-2025-08-07")
API_URL = os.getenv("API_URL", "http://yunwu.ai/v1")
API_KEY = os.getenv("API_KEY", "sk-yb4fxa2WVsZunr0G2OHz56QPjTHxgSnYBJD6GZE6EUYtK1ZN")

client = OpenAI(
    base_url = API_URL,
    api_key = API_KEY,
    timeout = 60,
    max_retries = 3,
)


def get_eval_generic(
    question: str,
    correct_answer: str,
    predict_answer: str,
    task: str = "correctness",
    max_tokens: int = 10000,
) -> str:
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
                f"Correct Answer: {correct_answer}\n"
                f"Predicted Answer: {predict_answer}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
        # messages = [
        #     {
        #         "role": "user",
        #         "content": "Compare the predicted answer with the correct answer for factual consistency. "
        #         "Score 0 – 5 (5 = perfect match, allowing synonyms/paraphrasing; 0 = completely inconsistent or hallucinated). "
        #         "Respond ONLY with a Python dict: {'score': integer}. No other text."
        #         "------\n"
        #         f"Question: {question}\n"
        #         "------\n"
        #         f"Correct Answer: {correct_answer}\n"
        #         "------\n"
        #         f"Predicted Answer: {predict_answer}"
        #     }
        # ]
    else:
        raise ValueError(f"Unsupported task: {task}")

    output_text = ""
    try:
        response = client.chat.completions.create(
            model = EVAL_MODEL_NAME,
            messages = messages,
            temperature = 0,
            max_tokens = max_tokens,
            timeout = 30,
        )
        print(f"response = {response}")

        output_text = response.choices[0].message.content
        print(f"output_text = {output_text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    return output_text


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", -1)
        return int(score)
    except SyntaxError as e:
        print(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return -1
    except ValueError as e:
        print(f"Value error parsing the review string: {e}. Review content: {review}")
        return -1
    except Exception as e:
        print(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return -1



def grade_single(question, correct_answer, predict_answer) -> float:
    scores = []
    for _ in range(3):
        review = get_eval_generic(question, correct_answer, predict_answer)
        score = parse_score(review)
        if score >= 0: # skip invalid score
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0


if __name__ == "__main__":
    question = "Offer a detailed interpretation of the video's message and imagery."
    correct_answer = "The video begins with a view of a bathroom, focusing on a mirror with a light fixture above it. The light fixture has four bulbs, all of which are illuminated. To the left of the mirror, there is a set of plastic drawers with various items on top. A person, partially visible and wearing dark clothing, is seen moving in the reflection of the mirror. The person steps closer to the light fixture and begins to adjust or replace one of the light bulbs, ensuring they are properly secured. After completing the task with the light fixture, the person steps back.\n\nThe scene then shifts to a different angle, showing a wooden cabinet with a countertop cluttered with various items, including bottles and containers. The same person, still partially visible and wearing dark clothing, enters the frame holding a broom. They begin to sweep the floor, moving the broom back and forth, focusing on cleaning the area near the cabinet.\n\nThe video continues with the person sweeping the floor, maintaining their focus on cleaning the area near the cabinet. The bathroom setting remains consistent, with the illuminated light fixture above the mirror and the cluttered countertop in view. The person diligently sweeps the floor, ensuring the area is clean and tidy."
    predict_answer = "The video begins with a view of a bathroom, illuminated by a row of light bulbs above a mirror. The room is cluttered with various items, including a basket filled with clothes and other a shelf holding miscellaneous objects. A person enters the frame from the left, walks towards the mirror, and then turns around to exit the room. The camera then then pans to reveal more of the bathroom, including a sink area with a cabinet underneath and shelves on the wall. The person then then picks up a broom and starts sweeping the floor, moving it back and forth across the room. The camera follows their the person's movements, capturing the sweeping action the entire room, including the corners and under the baseboards. The person continues continues to sweep diligently, ensuring that no part of the room is left missed. The video ends with the person still still still standing in the room, having the broom resting against the wall, and the room looking now clean and tidy."
    score = grade_single(question, correct_answer, predict_answer)
    print(f"Final score: {score}")



'''
curl --location --request POST 'http://yunwu.ai/v1/chat/completions' \
--header 'Accept: application/json' \
--header 'Authorization: Bearer sk-yb4fxa2WVsZunr0G2OHz56QPjTHxgSnYBJD6GZE6EUYtK1ZN' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model": "gpt-5-mini-2025-08-07",
  "messages": [
    {
      "role": "user",
      "content": "hello."
    }
  ],
  "max_tokens": 100
}'

'''