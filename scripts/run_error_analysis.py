from src.error_analysis import main, HumanConfig, DEFAULT_DATASET_NAME

if __name__ == "__main__":
    exp_name = "2) COMPLEX CONTEXT UPDATED: personal comprehensive + top25_20220628, refl3.-2438ea7c"
    config = HumanConfig(
        include_code=False,
        include_code_execution=False,
        include_code_statements=True,
        include_code_mismatches=False,
        include_rag=False,
        include_rag_statements=False,
        include_plans=False,
        include_all_in_one=False,
        statement_status_filter="both",
        judging_enabled=True
    )
    result = main(exp_name=exp_name, config=config, dataset_name=DEFAULT_DATASET_NAME)
    print(f"Analysis complete. Page ID: {result.get('page_id')}")
