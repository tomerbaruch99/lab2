"""
Uses Gemini to evaluate:
- RAG Answer vs. Right Answer
- Direct Answer vs. Right Answer

It produces per-question scores (factuality, relevance, completeness, confidence, correctness),
and saves CSV files into evaluation/.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict

import google.generativeai as genai
import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from tqdm.auto import tqdm

# ----------------- Config -----------------

API_KEYS_PATH = Path("../src/api_keys.json")  # adjust if needed
MODEL_RESPONSES_FOLDER = Path("model_responses")
EVALUATION_FOLDER = Path("evaluation")
EVALUATION_FOLDER.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL_NAME = "gemini-2.5-flash"
SLEEP_BETWEEN_CALLS = 5.0  # seconds, to avoid rate limits


# ----------------- Evaluation prompt -----------------

EVALUATION_PROMPT = """### Task Description:
An instruction (might include an input inside it), a response to evaluate, a reference answer that receives a score of 5, and a score rubric representing multiple evaluation criteria are provided.

1. Write specific and constructive feedback that assesses the response's quality strictly based on the given score rubrics below. If the response is more detailed or lengthy, this is not a disadvantage unless it includes off-topic or irrelevant content.
2. After writing feedback, provide a score between 1 and 5 for each evaluation criterion.
3. After feedback, and scores, provide an overall correctness score (Correct or Incorrect) if the response, in the context of a yes/no question, is correct.
4. Format your output as: "Feedback: {{feedback for each criterion}} [SCORE_FACTUALITY] {{score}} [SCORE_RELEVANCE] {{score}} [SCORE_COMPLETENESS] {{score}} [SCORE_CONFIDENCE] {{score}} [CORRECTNESS] {{Correct or Incorrect}}"
5. Please do not add any other opening, closing, or explanations. Include [SCORE_FACTUALITY], [SCORE_RELEVANCE], [SCORE_COMPLETENESS], [SCORE_CONFIDENCE], and [CORRECTNESS] in your output.

### The instruction to evaluate:
{instruction}

### Response to evaluate:
{response}

### Reference Answer (Score 5):
{reference_answer}

### Score Rubrics:
1. **Factuality**: Is the response correct, accurate, and factual based on the reference answer?
   - Score 1: Completely incorrect, inaccurate, and/or not factual.
   - Score 2: Mostly incorrect, inaccurate, and/or not factual.
   - Score 3: Somewhat correct, accurate, and/or factual.
   - Score 4: Mostly correct, accurate, and factual.
   - Score 5: Completely correct, accurate, and factual.

2. **Relevance**: Does the response stay focused on the instruction and provide relevant information without introducing unnecessary or off-topic content?
   - Score 1: Completely irrelevant to the instruction or question.
   - Score 2: Mostly irrelevant with some on-topic information.
   - Score 3: Somewhat relevant but includes some unnecessary information.
   - Score 4: Mostly relevant with little unnecessary information.
   - Score 5: Fully relevant and focused on the instruction.

3. **Completeness**: Does the response thoroughly cover all parts of the question or instruction without omitting important details?
   - Score 1: Completely incomplete, misses all key points.
   - Score 2: Misses most key points, partially complete.
   - Score 3: Addresses some key points but is incomplete in other aspects.
   - Score 4: Addresses most key points with minor omissions.
   - Score 5: Fully complete, addresses all key points directly.

4. **Confidence**: How confident is the response in providing accurate information based on the reference answer?
   - Score 1: Completely unsure or lacking confidence.
   - Score 2: Mostly unsure, indicates low confidence.
   - Score 3: Somewhat confident but lacks strong evidence.
   - Score 4: Mostly confident with some solid backing.
   - Score 5: Completely confident, well-supported by evidence.

### Feedback:"""

# ----------------- Setup Gemini evaluator -----------------

def load_api_keys(api_keys_path: Path) -> Dict[str, str]:
    with api_keys_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_evaluator(api_keys: Dict[str, str]):
    genai.configure(api_key=api_keys["GEMINI_API_KEY"])
    evaluator = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    return evaluator


evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT)]
)


# ----------------- Helpers -----------------

def load_test_set(file_path: str, rag_flag: bool) -> List[Dict]:
    """
    Load a single model_responses .parquet file and construct a list of examples
    compatible with the evaluation logic.

    rag_flag=True -> use 'RAG Answer'
    rag_flag=False -> use 'Direct Answer'
    """
    result_df = pd.read_parquet(file_path)
    testset = []
    for _, row in result_df.iterrows():
        question = row["Question"]
        answer = row["RAG Answer"] if rag_flag else row["Direct Answer"]
        ground_truth = row["Right Answer"]
        company = row["Company"]
        similarity_score = row["Similarity Score"]
        optimal_index = row["Optimal Index"] if "Optimal Index" in result_df.columns else None

        testset.append(
            {
                "question": question,
                "generated_answer": answer,
                "true_answer": ground_truth,
                "company": company,
                "similarity_score": similarity_score,
                "optimal_index": optimal_index,
            }
        )
    return testset


def extract_evaluation_metrics(eval_response_text: str) -> Dict:
    text = eval_response_text

    # feedback: everything before the first score marker
    feedback = text.split("[SCORE_FACTUALITY]")[0].strip()

    factuality_score = int(
        text.split("[SCORE_FACTUALITY]")[1].split("[SCORE_RELEVANCE]")[0].strip()
    )
    relevance_score = int(
        text.split("[SCORE_RELEVANCE]")[1].split("[SCORE_COMPLETENESS]")[0].strip()
    )
    completeness_score = int(
        text.split("[SCORE_COMPLETENESS]")[1].split("[SCORE_CONFIDENCE]")[0].strip()
    )
    confidence_score = int(
        text.split("[SCORE_CONFIDENCE]")[1].split("[CORRECTNESS]")[0].strip()
    )
    correctness = text.split("[CORRECTNESS]")[1].strip()

    return {
        "feedback": feedback,
        "factuality_score": factuality_score,
        "relevance_score": relevance_score,
        "completeness_score": completeness_score,
        "confidence_score": confidence_score,
        "correctness": correctness,
    }


def evaluate_and_save(testset, evaluator, test_name: str, save_path: Path, rag_flag: bool):
    save_path.mkdir(parents=True, exist_ok=True)
    evaluation_results = []

    mode_suffix = "_RAG" if rag_flag else "_Direct"

    for experiment in tqdm(testset, desc=f"Evaluating {test_name}{mode_suffix}"):
        evaluation_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_response = evaluator.generate_content(str(evaluation_prompt))
        time.sleep(SLEEP_BETWEEN_CALLS)

        eval_metrics = extract_evaluation_metrics(eval_response.text)

        # Parse similarity score list (stored as string, e.g. "[0.81, 0.72, ...]")
        try:
            sim_list = [
                float(x)
                for x in str(experiment["similarity_score"])[1:-1].split(", ")
                if x.strip() != ""
            ]
        except Exception:
            sim_list = []

        mean_sim = float(np.mean(sim_list)) if len(sim_list) > 0 else np.nan
        max_sim = float(np.max(sim_list)) if len(sim_list) > 0 else np.nan

        experiment.update(
            {
                "eval_factuality": eval_metrics["factuality_score"],
                "eval_relevance": eval_metrics["relevance_score"],
                "eval_completeness": eval_metrics["completeness_score"],
                "eval_confidence": eval_metrics["confidence_score"],
                "eval_correctness": eval_metrics["correctness"],
                "eval_feedback": eval_metrics["feedback"],
                "mean_similarity_score": mean_sim,
                "max_similarity_score": max_sim,
            }
        )
        evaluation_results.append(experiment)

    df_out = pd.DataFrame(evaluation_results)
    out_file = save_path / f"{test_name}{mode_suffix}_evaluation.csv"
    df_out.to_csv(out_file, index=False)
    print(f"[OK] Saved evaluation file: {out_file}")


def main():
    api_keys = load_api_keys(API_KEYS_PATH)
    evaluator = init_evaluator(api_keys)

    for file_name in tqdm(os.listdir(MODEL_RESPONSES_FOLDER), desc="Scanning model_responses"):
        if not file_name.endswith(".parquet"):
            continue

        file_path = MODEL_RESPONSES_FOLDER / file_name
        test_name = os.path.splitext(file_name)[0]

        for rag_flag in [True, False]:
            testset = load_test_set(str(file_path), rag_flag=rag_flag)
            evaluate_and_save(testset, evaluator, test_name, EVALUATION_FOLDER, rag_flag=rag_flag)


if __name__ == "__main__":
    main()
