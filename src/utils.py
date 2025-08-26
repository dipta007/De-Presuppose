import json
import re

from loguru import logger
from openai import OpenAI
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def get_binary_metrics(args, records):
    label_to_id = {"supported": 1, "not_supported": 0}
    id_to_label = {v: k for k, v in label_to_id.items()}
    y_true = [label_to_id.get(record["label"].lower(), 0) for record in records]
    y_pred = [label_to_id.get(record["reasoner_decision"].lower(), 0) for record in records]

    acc = accuracy_score(y_true, y_pred) * 100
    bacc = balanced_accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="binary") * 100
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class_acc = {}
    for i, label in enumerate([0, 1]):
        per_class_acc[id_to_label[label]] = (conf_matrix[i, i] / max(conf_matrix[i, :].sum(), 1)) * 100

    return dict(
        acc=acc,
        bacc=bacc,
        f1=f1,
        conf_matrix=conf_matrix.tolist(),
        per_class_acc=per_class_acc,
    )


@retry(wait=wait_random_exponential(min=1, max=600), stop=stop_after_attempt(7))
def completion_with_backoff(client, **kwargs):
    try:
        ret = client.chat.completions.create(**kwargs)
        return ret
    except Exception as e:
        logger.error(e)
        raise e


def safe_parse_json(model_answer, hidden_reasoning=None):
    try:
        return json.loads(model_answer)
    except Exception:
        model_answer = model_answer.strip()
        if hidden_reasoning:
            return dict(reasoning=hidden_reasoning, decision=model_answer)
        else:
            return dict(reasoning="Failed to parse JSON", decision=model_answer)


def get_generation_arguments(model):
    GENERATION_ARGUMENTS = {
        "QA": dict(
            temperature=0.0,
            frequency_penalty=1.02,
            max_completion_tokens=2048,
        ),
        r"meta-llama\/Llama-.*": dict(
            temperature=0.0,
            max_completion_tokens=4096,
        ),
        r"Qwen\/Qwen2\.5-.*-Instruct": dict(
            temperature=0.0,
            max_completion_tokens=8192,
        ),
        "gpt-4o.*": dict(
            temperature=0.0,
            max_completion_tokens=4096,
        ),
        # thinking - https://huggingface.co/Qwen/Qwen3-32B#best-practices
        "Qwen/Qwen3-.*@think": dict(
            temperature=0.6,
            top_p=0.95,
            max_completion_tokens=14336,  # 16384 - 2048
            presence_penalty=1,
            extra_body=dict(
                min_p=0.0,
                top_k=20,
                chat_template_kwargs=dict(
                    enable_thinking=True,
                ),
            ),
        ),
        # non-thinking - https://huggingface.co/Qwen/Qwen3-32B#best-practices
        "Qwen/Qwen3-.*@nothink": dict(
            temperature=0.7,
            top_p=0.8,
            max_completion_tokens=14336,  # 16384 - 2048
            presence_penalty=1,
            extra_body=dict(
                min_p=0.0,
                top_k=20,
                chat_template_kwargs=dict(
                    enable_thinking=False,
                ),
            ),
        ),
        "Qwen/Qwen3-.*@temp0": dict(
            temperature=0.0,
            max_completion_tokens=2048,
            extra_body=dict(
                chat_template_kwargs=dict(
                    enable_thinking=False,
                ),
            ),
        ),
        # followed their Guidelines: https://huggingface.co/Qwen/QwQ-32B#usage-guidelines
        "Qwen/QwQ-32B.*": dict(
            temperature=0.6,
            top_p=0.95,
            # presence_penalty=1.02,
            max_completion_tokens=14336,  # 16384 - 2048
            # TODO: remove this
            # max_completion_tokens=8192,  # 16384 - 2048
            extra_body=dict(
                top_k=40,
                stop=["<|im_start|>", "<|im_end|>"],
                repeat_penalty=1,
            ),
        ),
        "deepseek-ai/DeepSeek-R1-Distill-.*": dict(
            temperature=0.6,
            top_p=0.95,
            max_completion_tokens=14336,  # 16384 - 2048
        ),
        "o.-mini": dict(
            reasoning_effort="high",
            max_completion_tokens=14336,  # 16384 - 2048
        ),
        "gemini-2.5-pro-exp-03-25": dict(
            temperature=0.0,
            max_completion_tokens=14336,  # 16384 - 2048
        ),
        "default": dict(
            temperature=0.0,
            max_completion_tokens=2048,
        ),
    }
    COMMON_ARGUMENTS = dict(
        seed=42,
    )
    gen_args = GENERATION_ARGUMENTS["default"]
    found = False
    for key, value in GENERATION_ARGUMENTS.items():
        if re.match(key, model):
            gen_args = value
            found = True
            break
    if not found:
        logger.warning(f"No generation arguments found for model: {model}")
    gen_args.update(COMMON_ARGUMENTS)
    logger.debug(f"Generation arguments for {model}: {gen_args}")
    return gen_args


GPT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
    "o3-mini",
    "o4-mini",
]

####################
# Initialize clients
####################
try:
    qa_client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="EMPTY",
    )
    qa_models = [model.id for model in qa_client.models.list()]
except Exception:
    qa_client = []
    qa_models = []

try:
    gpt_client = OpenAI()
    gpt_models = [model.id for model in gpt_client.models.list()]
except Exception:
    gpt_client = []
    gpt_models = []

try:
    oss_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    oss_models = [model.id for model in oss_client.models.list()]
except Exception:
    oss_client = []
    oss_models = []


def get_client(model):
    if model in gpt_models:
        return gpt_client
    elif model in qa_models:
        return qa_client
    elif model in oss_models:
        return oss_client
    else:
        raise ValueError(f"Model {model} not found")
