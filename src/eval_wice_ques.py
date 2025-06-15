import argparse
import glob
import json
import os
import time

import jsonlines
import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from openai import OpenAI
from src.reasoner.utils import get_generation_arguments
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm


def get_subclaims():
    claim2subclaims = {}

    with jsonlines.open("data/wice/test.jsonl") as reader:
        for line in reader:
            claim2subclaims[line["claim"]] = [(x["claim"], x["label"]) for x in line["subclaims"]]

    return claim2subclaims


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(7))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


oss_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

COVERAGE_PROMPT = """\
Given a claim, evidence, and a list of questions, analyze whether the questions collectively are sufficient to verify or refute the entire claim.

### Instructions
- We are looking for coverage of the claim not completeness of the questions. So, if some questions are not relevant to the claim, that's fine. But if the relevant questions do not cover the whole claim, then the coverage is not good.
- The question does not need to ask the specific claim explicitly. If answering the question would verify the claim, then it covers the claim.
- It is possible that multiple questions together cover the claim. It is not necessary that the claim is covered by a single question.
- If a question and claim refer to similar, but non-identical concepts, use the provided evidence to determine whether the question and claim are referring to the same concept or not. For example, the claim may refer to "the machine learning technique," while the question may ask about "the supervised learning technique." Because the questions were generated based on the provided evidence, consider this evidence when determining your final answer.
- Begin by providing 1-2 sentences explaining your reasoning for the coverage of the claim.
- Afterward, output yes if the questions cover the claim completely, or no if they do not.
- Structure your final response into two sections:
    - EXPLANATION: (your reasoning in 1-2 sentences)
    - ANSWER: (Yes if the questions cover the claim completely, or No if they do not)

### Evidence
{evidence}

### Claim
{claim}

### Questions
{questions}
"""

ACCURACY_PROMPT = """\
Given a claim, its verification label (Supported or Not Supported), and a list of question–answer pairs meant to verify the claim, assess whether the answers are accurate and sufficient to justify the verification label.

### Instructions
- We are looking for accuracy of the answers not completeness of the questions. So, if some questions are not relevant to the claim, that's fine. But if the answers verify the claim's label correctly, then the accuracy is good.
- Begin by providing 1-2 sentences explaining your reasoning for the accuracy of the answers.
- Afterward, output yes if the answers are accurate and sufficient to justify the verification label, or no if they are not.
- Structure your final response into two sections:
    - EXPLANATION: (your reasoning in 1-2 sentences)
    - ANSWER: (Yes if the answers are accurate and sufficient to justify the verification label, or No if they are not)

### Claim
{claim}

### Verification Label
{label}

### Questions
{questions}

### Answers
{answers}
"""


def get_coverage(args, subclaim, questions, evidence):
    formatted_questions = "\n".join([f"- {question}" for i, question in enumerate(questions)])
    prompt = COVERAGE_PROMPT.format(claim=subclaim, questions=formatted_questions, evidence=evidence)
    logger.debug(f"Coverage Prompt: {prompt}")
    response = completion_with_backoff(
        client=oss_client,
        model=args.model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **get_generation_arguments(args.raw_model_id),
    )
    generation = response.choices[0].message.content
    logger.debug(f"Coverage Generation: {generation}")
    return prompt, generation


def get_accuracy(args, subclaim, label, questions, answers):
    formatted_questions = "\n".join([f"{i + 1}. {question}" for i, question in enumerate(questions)])
    formatted_answers = "\n".join([f"{i + 1}. {answer}" for i, answer in enumerate(answers)])
    prompt = ACCURACY_PROMPT.format(claim=subclaim, label=label, questions=formatted_questions, answers=formatted_answers)
    logger.debug(f"Accuracy Prompt: {prompt}")
    response = completion_with_backoff(
        client=oss_client,
        model=args.model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **get_generation_arguments(args.raw_model_id),
    )
    generation = response.choices[0].message.content
    logger.debug(f"Accuracy Generation: {generation}")
    return prompt, generation


def run(args, claim, subclaims, questions, answers, evidence):
    record = dict(
        claim=claim,
        subclaims=subclaims,
        questions=questions,
        answers=answers,
        subclaim2coverage={},
        subclaim2accuracy={},
        ques_coverage=None,
        ans_accuracy=None,
    )

    coverage = 0
    accuracy = 0
    for subclaim, label in subclaims:
        try:
            prompt, generation = get_coverage(args, subclaim, questions, evidence)
            output = generation.split("ANSWER:")[-1].strip()
            coverage += 1 if output.lower() == "yes" else 0
        except:
            logger.error(f"Error parsing generation: {generation}")
            output = "No"
        record["subclaim2coverage"][subclaim] = dict(
            subclaim=subclaim,
            prompt=prompt,
            generation=generation,
            output=output,
        )

        if len(answers) == 0:
            logger.warning(f"No answers for subclaim: {subclaim}")
            continue
        try:
            prompt, generation = get_accuracy(args, subclaim, label, questions, answers)
            output = generation.split("ANSWER:")[-1].strip()
            accuracy += 1 if output.lower() == "yes" else 0
        except:
            logger.error(f"Error parsing generation: {generation}")
            output = "No"
        record["subclaim2accuracy"][subclaim] = dict(
            subclaim=subclaim,
            prompt=prompt,
            generation=generation,
            output=output,
        )

    record["ques_coverage"] = coverage / len(subclaims)
    record["ans_accuracy"] = accuracy / len(subclaims)
    return record


def eval_wice(args, run_num):
    claim2subclaims = get_subclaims()
    func_args = []
    with jsonlines.open(args.eval_gen_path) as reader:
        for line in reader:
            subclaims = claim2subclaims[line["claim"]]
            # subclaims = [x[0] for x in subclaims]
            # subclaims = list(set(subclaims))
            if "Q_questions" not in line:
                logger.warning(f"No questions for claim: {line['claim']}")
                raise Exception(f"No questions for claim: {line['claim']}")
            questions = line["Q_questions"]
            answers = line.get("A_answer", [])
            evidence = line.get("evidence", "")
            func_args.append((args, line["claim"], subclaims, questions, answers, evidence))

    #############################
    # Run the reasoner parallelly
    #############################
    start = time.time()
    records = Parallel(n_jobs=args.num_of_workers, prefer="threads")(
        delayed(run)(*func_args) for i, func_args in tqdm(enumerate(func_args), total=len(func_args), desc="Processing")
    )
    time_taken = time.time() - start
    logger.info(f"Time taken: {time_taken}")

    ##################
    # Save the records
    ##################
    out_file = f"{args.output_dir}/coverage_results_{run_num}.jsonl"
    with jsonlines.open(out_file, "w") as f:
        for record in records:
            f.write(record)

    ##################
    # Save the metrics
    ##################
    metrics = json.load(open(f"{args.output_dir}/metrics_{run_num}.json"))

    metrics["ques_coverage"] = [record["ques_coverage"] for record in records]
    metrics["ques_coverage_mean"] = sum(metrics["ques_coverage"]) / len(metrics["ques_coverage"])
    metrics["ques_coverage_mean"] *= 100

    metrics["ans_accuracy"] = [record["ans_accuracy"] for record in records]
    metrics["ans_accuracy_mean"] = sum(metrics["ans_accuracy"]) / len(metrics["ans_accuracy"])
    metrics["ans_accuracy_mean"] *= 100

    metrics_file = f"{args.output_dir}/metrics_{run_num}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics {run_num}: {json.dumps(metrics, indent=2)}")
    print(f"Metrics {run_num}: {json.dumps(metrics, indent=2)}")
    return True


def aggregate_metrics(output_dir, num_of_runs):
    metrics_files = [f"{output_dir}/metrics_{i}.json" for i in range(num_of_runs)]
    aggregated_metrics = json.load(open(f"{output_dir}/metrics_aggregated.json"))
    for key in ["ques_coverage_mean", "ans_accuracy_mean"]:
        aggregated_metrics[key] = []

    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            current_metrics = json.load(f)
            for key in ["ques_coverage_mean", "ans_accuracy_mean"]:
                aggregated_metrics[key].append(current_metrics[key])

    # average the metrics
    for key in ["ques_coverage_mean", "ans_accuracy_mean"]:
        aggregated_metrics[key] = [np.mean(aggregated_metrics[key]), np.std(aggregated_metrics[key])]

    # save the aggregated metrics
    with open(f"{output_dir}/metrics_aggregated.json", "w") as f:
        json.dump(aggregated_metrics, f, indent=2)


if __name__ == "__main__":
    logger.remove()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_gen_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-32B@temp0")
    parser.add_argument("--num_of_workers", type=int, default=64)
    args = parser.parse_args()

    args.raw_model_id = args.model_id
    args.model_id = args.model_id.split("@")[0]
    ROOT_DIRS = [
        "outputs/bionli",
        "outputs/wice",
        "outputs/fever",
    ]
    error_exps = []
    for ROOT_DIR in ROOT_DIRS:
        dirs = os.listdir(ROOT_DIR)
        dirs = [x for x in dirs if os.path.isdir(f"{ROOT_DIR}/{x}")]
        for exp in dirs:
            error_flag = False
            records_files = glob.glob(f"{ROOT_DIR}/{exp}/records_[0-9]*.jsonl", recursive=False)
            num_of_runs = len(records_files)
            for i, path_to_record in enumerate(records_files):
                path_to_coverage_results = path_to_record.replace("records_", "coverage_results_")
                if os.path.exists(path_to_coverage_results):
                    logger.info(f"Skipping {path_to_record} because it already exists")
                    continue
                args.eval_gen_path = path_to_record
                args.output_dir = os.path.dirname(args.eval_gen_path)
                print(json.dumps(args.__dict__, indent=4))
                try:
                    eval_wice(args, i)
                except Exception as e:
                    print(e)
                    logger.error(f"Error evaluating {path_to_record}: {e}")
                    error_exps.append(f"{exp}")
                    error_flag = True
                    continue

                if not error_flag and num_of_runs > 0:
                    aggregate_metrics(f"{ROOT_DIR}/{exp}", num_of_runs)

        print(f"Error experiments: {error_exps}")
