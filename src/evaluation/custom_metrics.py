import pandas as pd


def run_golden_tests(self_rag_app, golden_tests: list[dict]) -> pd.DataFrame:
    """Run the Self-RAG app on a golden test set and return an evaluation dataframe."""
    eval_rows = []

    for test in golden_tests:
        result = self_rag_app.invoke({
            "question": test["question"],
            "chat_history": [],
            "retry_count": 0
        })

        docs = result.get("retrieved_docs", [])

        eval_rows.append({
            "question": test["question"],
            "answer": result.get("final_answer", ""),
            "contexts": [doc.page_content for doc in docs],
            "ground_truth": test["ground_truth"],
            "expected_docs": test["expected_docs"],
            "retrieved_files": [doc.metadata.get("file_name") for doc in docs],
            "context_score": result.get("context_score"),
            "faithfulness_score": result.get("faithfulness_score"),
            "retry_count": result.get("retry_count"),
            "classification_reason": result.get("classification_reason"),
        })

    return pd.DataFrame(eval_rows)


def add_retrieval_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Add Hit Rate@5 and All Expected Found@5 to an evaluation dataframe."""
    eval_df = eval_df.copy()

    def hit_rate(row):
        expected = set(row["expected_docs"])
        retrieved = set(row["retrieved_files"])
        return int(bool(expected.intersection(retrieved)))

    def all_expected_found(row):
        expected = set(row["expected_docs"])
        retrieved = set(row["retrieved_files"])
        return int(expected.issubset(retrieved))

    eval_df["hit_rate_at_5"] = eval_df.apply(hit_rate, axis=1)
    eval_df["all_expected_found_at_5"] = eval_df.apply(all_expected_found, axis=1)

    return eval_df


def summarize_metrics(eval_df: pd.DataFrame) -> dict:
    """Create a compact metric summary."""
    summary = {
        "total_questions": len(eval_df),
        "hit_rate_at_5": eval_df["hit_rate_at_5"].mean() if "hit_rate_at_5" in eval_df else None,
        "all_expected_found_at_5": eval_df["all_expected_found_at_5"].mean() if "all_expected_found_at_5" in eval_df else None,
        "avg_context_score": eval_df["context_score"].mean() if "context_score" in eval_df else None,
        "avg_faithfulness_score": eval_df["faithfulness_score"].mean() if "faithfulness_score" in eval_df else None,
    }
    return summary
