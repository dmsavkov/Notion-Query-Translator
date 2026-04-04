from typing import Any, Dict


def format_lifecycle_result(result: Dict[str, Dict[str, Any]]) -> str:
    if not result:
        return "No results produced."

    if "user_request" in result:
        ordered_items = [("user_request", result["user_request"])]
    else:
        ordered_items = sorted(result.items(), key=lambda item: item[0])

    blocks = []
    for task_id, task_result in ordered_items:
        solution_run = task_result.get("solution_run") or {}
        if not isinstance(solution_run, dict):
            solution_run = {}

        passed = bool(task_result.get("passed", solution_run.get("passed", False)))
        status = "PASS" if passed else "FAIL"
        output = str(
            task_result.get("execution_output")
            or task_result.get("output")
            or "(no output)"
        )
        final_code = str(
            task_result.get("final_code")
            or task_result.get("generated_code")
            or ""
        )
        error = str(task_result.get("error") or "")

        lines = [f"[{task_id}] {status}"]
        if error:
            lines.append(f"ERROR: {error}")
        lines.append("Execution Output:")
        lines.append(output)

        if final_code:
            lines.append("Generated Code:")
            lines.append(final_code)

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
