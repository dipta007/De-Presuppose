import os
from loguru import logger

reasoners = [
    # "Qwen/Qwen3-32B@think",
    "Qwen/QwQ-32B",
    # "o4-mini",
]

questions = [
    # "Qwen/Qwen3-32B@think",
    "Qwen/QwQ-32B",
    # "o4-mini",
    None,
]

answers = [
    # "Qwen/Qwen3-32B@think",
    "Qwen/QwQ-32B",
    # "o4-mini",
    None,
]

depreposition_types = [
    False,
    True,
]

verifier_prompts = [
    "sg1",
    "sg2",
    "minicheck",
]

dataset_paths = []
ROOT_DIRS = [
    "data/bionli",
    # "data/wice",
    # "data/fever",
]
for ROOT_DIR in ROOT_DIRS:
    for root, dirs, files in os.walk(ROOT_DIR):
        for f in files:
            if "_300" not in f:
                continue
            if f.endswith(".csv") and "tmp" not in f:
                dataset_paths.append(f"{root}/{f}")
dataset_paths = dataset_paths[::-1]
print("=" * 100)
for d in dataset_paths:
    print(d)
print("=" * 100)
# exit()


def run_cmd(cmd):
    logger.info(cmd)
    ret = os.system(cmd)
    if ret != 0:
        logger.error(f"failed: {cmd}")
    else:
        logger.success(f"success: {cmd}")


for dataset_path in dataset_paths:
    for reasoner in reasoners:
        for question in questions:
            for answer in answers:
                for verifier_prompt in verifier_prompts:
                    # if there is answer, there needs to be a question
                    if not question and answer:
                        continue
                    # if there is question and answer, they need to be the same
                    if question and answer and question != answer:
                        continue
                    # if there is reasoner, question and answer, they need to be the same
                    if reasoner and question and answer and len(set([reasoner, question, answer])) != 1:
                        continue
                    # if there is reasoner, question, they need to be the same
                    if reasoner and question and reasoner != question:
                        continue

                    for depreposition_type in depreposition_types:
                        cmd = f"python -u -m src.reasoner --reasoner_model_id {reasoner} --dataset_path {dataset_path} --verifier_prompt {verifier_prompt} --num_of_runs 3"

                        if question:
                            cmd += f" --question_model_id {question}"

                        if answer:
                            cmd += f" --answer_model_id {answer}"

                        # if there is no question, there is no need to run depreposition
                        if depreposition_type and question is None:
                            continue

                        if depreposition_type:
                            cmd += " --depresupposition"

                        # for openai models, less number of parallel runs
                        if "o4-mini" in reasoner or "o4-mini" in question or "o4-mini" in answer:
                            cmd += " --num_of_workers 2"
                        else:
                            cmd += " --num_of_workers 64"

                        run_cmd(cmd)
