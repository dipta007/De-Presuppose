import argparse
import json
import os
import random
import time
import warnings

import jsonlines
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm, trange

from src.prompts.qa.decomposition import PQA_QUESTION2ANSWER_PROMPT, PROMPT_FROM_WICE
from src.prompts.qa.depresupposition import DEPRESUPPOSITION_PROMPT
from src.prompts.qa.utils import generated_text2questions
from src.prompts.verifier.minicheck import MINICHECK_VERIFIER_PROMPT_BINARY
from src.prompts.verifier.sg1 import SG1_VERIFIER_PROMPT_BINARY
from src.prompts.verifier.sg2 import SG2_VERIFIER_PROMPT_BINARY
from src.utils import completion_with_backoff, get_client, get_generation_arguments, get_binary_metrics, safe_parse_json

warnings.filterwarnings("ignore")

random.seed(42)

PROMPT_MAPPING = dict(
    sg1=SG1_VERIFIER_PROMPT_BINARY,
    sg2=SG2_VERIFIER_PROMPT_BINARY,
    minicheck=MINICHECK_VERIFIER_PROMPT_BINARY,
)


def do_depresupposition(args, question, record) -> list[str]:
    prompt = DEPRESUPPOSITION_PROMPT.format(question=question)
    logger.debug(f"depresupposition Prompt: {prompt}")
    response = completion_with_backoff(
        client=get_client(args.depresupposition_model_id),
        model=args.depresupposition_model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **get_generation_arguments(args.raw_depresupposition_model_id),
    )
    generation = response.choices[0].message.content
    logger.debug(f"depresupposition Generation: {generation}")

    record[f"QD_prompt_{question}"] = prompt
    record[f"QD_generation_{question}"] = generation
    if "</think>" in generation:
        generation = generation.split("</think>")[1]
        generation = generation.strip()

    questions = generated_text2questions(generation)
    logger.debug(f"depresupposition Questions: {'\n'.join(questions)}")

    record[f"QD_questions_{question}"] = questions
    return questions


def get_questions(args, claim, evidence, record) -> list[str]:
    prompt = PROMPT_FROM_WICE.format(claim=claim, evidence=evidence)
    logger.debug(f"Question Generation Prompt: {prompt}")
    response = completion_with_backoff(
        client=get_client(args.question_model_id),
        model=args.question_model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **get_generation_arguments(args.raw_question_model_id),
    )
    generation = response.choices[0].message.content
    logger.debug(f"Question Generation Generation: {generation}")

    record["Q_prompt"] = prompt
    record["Q_generation"] = generation
    if "</think>" in generation:
        generation = generation.split("</think>")[1]
        generation = generation.strip()

    questions = generated_text2questions(generation)
    logger.debug(f"Questions: {'\n'.join(questions)}")

    if args.depresupposition:
        depresupposition_questions = []
        for question in questions:
            depresupposition_questions.append(question)
            depresupposition_questions.extend(do_depresupposition(args, question, record))
        questions = list(set(depresupposition_questions))

    record["Q_questions"] = questions
    return questions


def get_answers(args, evidence, questions, record) -> list[str]:
    record["A_prompt"] = []
    record["A_answer"] = []
    record["A_generation"] = []

    def get_answer(question):
        prompt = PQA_QUESTION2ANSWER_PROMPT.format(document=evidence, question=question)
        logger.debug(f"Question Answer Prompt: {prompt}")
        response = completion_with_backoff(
            client=get_client(args.answer_model_id),
            model=args.answer_model_id,
            messages=[
                {"role": "user", "content": prompt},
            ],
            **get_generation_arguments(args.raw_answer_model_id),
        )
        generation = response.choices[0].message.content
        logger.debug(f"Question Answer Generation: {generation}")

        record["A_prompt"].append(prompt)
        record["A_generation"].append(generation)
        if "</think>" in generation:
            generation = generation.split("</think>")[1]
            generation = generation.strip()

        answer = generation.strip()
        if not answer:
            logger.error(f"Failed to parse answer: {generation}")
            answer = "No answer found"

        record["A_answer"].append(answer)
        return answer

    answers = [get_answer(q) for q in questions]
    return answers


def get_evidence(args, row, df):
    return row["Gold"]


def run(args, row, df):
    claim = row["Query"]
    evidence = get_evidence(args, row, df)
    verification_evidence = evidence

    record = dict(
        claim=claim,
        evidence=evidence,
        verification_evidence=verification_evidence,
        label=row["Label"],
    )

    PROMPTS = PROMPT_MAPPING[args.verifier_prompt]
    prompt = ""
    if args.question_model_id and args.answer_model_id:
        prompt = PROMPTS["given_evidence_questions_answers"]
    elif args.question_model_id:
        prompt = PROMPTS["given_evidence_questions"]
    else:
        prompt = PROMPTS["given_evidence"]

    kwargs = dict(CLAIM=claim, EVIDENCE=verification_evidence)
    #####################
    # Claim --> Questions
    #####################
    questions = []
    if args.question_model_id:
        questions = get_questions(args, claim, evidence, record)

    #######################
    # Questions --> Answers
    #######################
    answers = []
    if args.answer_model_id:
        answers = get_answers(args, evidence, questions, record)

    questions = [f"{i + 1}. {q}" for i, q in enumerate(questions)]
    questions = "\n".join(questions)
    record["questions"] = questions
    kwargs["QUESTIONS"] = questions

    answers = [f"{i + 1}. {a}" for i, a in enumerate(answers)]
    answers = "\n".join(answers)
    record["answers"] = answers
    kwargs["ANSWERS"] = answers

    logger.debug(f"Questions: {questions}")
    logger.debug(f"Answers: {answers}")

    prompt = prompt.format(**kwargs)
    logger.debug(f"Reasoner Prompt: {prompt}")
    record["reasoner_prompt"] = prompt

    ###################################################
    # Claim, Evidence, Questions, Answers --> Reasoning
    ###################################################
    response = completion_with_backoff(
        client=get_client(args.reasoner_model_id),
        model=args.reasoner_model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **get_generation_arguments(args.raw_reasoner_model_id),
    )
    record["prompt_tokens"] = response.usage.prompt_tokens
    record["completion_tokens"] = response.usage.completion_tokens
    record["total_tokens"] = response.usage.total_tokens

    generation = response.choices[0].message.content
    logger.debug(f"Reasoner Response: {generation}")
    record["reasoner_generation"] = generation

    hidden_reasoning = generation
    if "</think>" in generation:
        hidden_reasoning = generation.split("</think>")[0]
        generation = generation.split("</think>")[1]
        generation = generation.strip()

    out = safe_parse_json(generation, hidden_reasoning)
    if out is None:
        logger.error("Failed to parse JSON")
        logger.error(f"Reasoner Response: {generation}")
        out = dict(reasoning="Failed to parse JSON", decision="UNKNOWN")

    record["reasoner_reasoning"] = out.get("reasoning")
    record["reasoner_decision"] = out.get("decision").strip()
    return record


def get_file_name(args):
    file_names = []

    reasoner_model_id = args.raw_reasoner_model_id.replace("/", "-")
    file_names.append(reasoner_model_id)

    question_model_id = args.raw_question_model_id.replace("/", "-") if args.question_model_id else "None"
    file_names.append(question_model_id)

    answer_model_id = args.raw_answer_model_id.replace("/", "-") if args.answer_model_id else "None"
    file_names.append(answer_model_id)

    if args.depresupposition:
        file_names.append("dep")

    return args.verifier_prompt + "/" + "_".join(file_names)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/bionli/bionli_300.csv", help="Path to the dataset")
    parser.add_argument("--reasoner_model_id", type=str, default="o4-mini", help="LLM to use for reasoning")
    parser.add_argument("--question_model_id", type=str, default=None, help="LLM to use for question generation")
    parser.add_argument("--answer_model_id", type=str, default=None, help="LLM to use for answer generation")
    parser.add_argument("--depresupposition", action="store_true", help="Enable depresupposition", default=False)
    parser.add_argument("--verifier_prompt", type=str, required=True, help="Verifier prompt to use")
    parser.add_argument("--num_of_workers", type=int, default=16, help="Number of workers to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging", default=False)
    parser.add_argument("--force", action="store_true", help="Overwrite existing results", default=False)
    parser.add_argument("--num_of_runs", type=int, default=3, help="Number of trials")
    args = parser.parse_args()

    if args.answer_model_id and not args.question_model_id:
        raise ValueError("Question model is required when answer model is provided")

    # process to extract the model id
    # case for hybrid models: "Qwen/Qwen3-32B@think" or "Qwen/QwQ-32B@nothink"
    if args.reasoner_model_id:
        args.raw_reasoner_model_id = args.reasoner_model_id
        args.reasoner_model_id = args.reasoner_model_id.split("@")[0]
    if args.question_model_id:
        args.raw_question_model_id = args.question_model_id
        args.question_model_id = args.question_model_id.split("@")[0]
    if args.answer_model_id:
        args.raw_answer_model_id = args.answer_model_id
        args.answer_model_id = args.answer_model_id.split("@")[0]

    if args.question_model_id and args.raw_question_model_id and args.depresupposition:
        args.raw_depresupposition_model_id = args.raw_question_model_id
        args.depresupposition_model_id = args.question_model_id

    args.output_dir = os.path.join("outputs", os.path.basename(os.path.dirname(args.dataset_path)), get_file_name(args))
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.debug:
        logger.remove()
        logger.add(
            f"{args.output_dir}/run.log",
            colorize=False,
            format="<cyan>{module:->16s}</cyan>.<blue>{line:04d}</blue> | <level>{level:8s}</level> | <level>{message}</level>",
            level="DEBUG",
            mode="w",
        )
    logger.info("Arguments:")
    logger.info(json.dumps(args.__dict__, indent=2))
    return args


def main(args, run_num):
    df = pd.read_csv(args.dataset_path)

    def convert_label(row):
        if row["Label"].lower() == "entailment":
            return "SUPPORTED"
        elif row["Label"].lower() == "contradiction":
            return "NOT_SUPPORTED"
        else:
            return row["Label"]

    df["Label"] = df.apply(convert_label, axis=1)

    if args.debug:
        df = df.sample(n=20)

    #############################
    # Run the reasoner parallelly
    #############################
    start = time.time()
    records = Parallel(n_jobs=args.num_of_workers, prefer="threads")(
        delayed(run)(args, row, df) for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing")
    )
    time_taken = time.time() - start
    logger.info(f"Time taken: {time_taken}")

    ##################
    # Save the records
    ##################
    if not args.debug:
        out_file = f"{args.output_dir}/records_{run_num}.jsonl"
        with jsonlines.open(out_file, "w") as f:
            for record in records:
                f.write(record)

    ##################
    # Save the metrics
    ##################
    if not args.debug:
        metrics = get_binary_metrics(args, records)
        metrics["args"] = args.__dict__
        metrics["time_taken"] = time_taken
        metrics_file = f"{args.output_dir}/metrics_{run_num}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info(f"Metrics {run_num}: {json.dumps(metrics, indent=2)}")
    print(f"Metrics {run_num}: {json.dumps(metrics, indent=2)}")


def aggregate_metrics(args, num_of_runs):
    metrics_files = [f"{args.output_dir}/metrics_{i}.json" for i in range(num_of_runs)]
    metrics = {}
    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            current_metrics = json.load(f)
            for key in ["acc", "bacc", "f1"]:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(current_metrics[key])
            for key in ["per_class_acc"]:
                if key not in metrics:
                    metrics[key] = {}
                for k, v in current_metrics[key].items():
                    if k not in metrics[key]:
                        metrics[key][k] = []
                    metrics[key][k].append(v)

    # average the binary metrics
    for k in ["acc", "bacc", "f1"]:
        metrics[k] = [np.mean(metrics[k]), np.std(metrics[k])]
    for k in ["per_class_acc"]:
        keys = list(metrics[k].keys())
        for kk in keys:
            metrics[k][kk] = [np.mean(metrics[k][kk]), np.std(metrics[k][kk])]

    # save the aggregated metrics
    with open(f"{args.output_dir}/metrics_aggregated.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    for run_num in trange(args.num_of_runs, desc="Running different runs"):
        if os.path.exists(f"{args.output_dir}/records_{run_num}.jsonl") and os.path.exists(f"{args.output_dir}/metrics_{run_num}.json") and not args.force:
            logger.info(f"Skipping {args.output_dir}/records_{run_num}.jsonl because it already exists")
            continue
        main(args, run_num)
    aggregate_metrics(args, args.num_of_runs)


# ? for reasoner
# ! vllm serve Qwen/QwQ-32B                             --max-model-len 32768   --port 8000 -tp 2
# ! vllm serve Qwen/Qwen3-32B                           --max-model-len 32768   --port 8000 -tp 2
