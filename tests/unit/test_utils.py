import pytest
import os
import shutil
from pathlib import Path
from src.all_functionality import build_general_info, load_eval_tasks
from run_pipeline import generate_thread_id

@pytest.mark.unit
def test_generate_thread_id():
    tid = generate_thread_id("task1")
    assert tid.startswith("task1_")
    assert len(tid) > 10
    
    tid_no_prefix = generate_thread_id()
    assert len(tid_no_prefix) == 14  # YYYYMMDDHHMMSS

@pytest.mark.unit
def test_build_general_info():
    prompt = "Add task A"
    context = "API context"
    plan = "1. Call API"
    
    info = build_general_info(prompt, context, plan)
    
    assert f"<user_request>\n{prompt}\n</user_request>" in info
    assert f"<api_context>\n{context}\n</api_context>" in info
    assert f"<request_plan>\n{plan}\n</request_plan>" in info
    assert "<reminder>" in info

@pytest.mark.unit
def test_load_eval_tasks(tmp_path):
    # Create dummy eval directory structure
    evals_dir = tmp_path / "evals"
    complex_dir = evals_dir / "complex"
    complex_dir.mkdir(parents=True)
    
    (evals_dir / "simple_task.yaml").write_text("query: simple", encoding="utf-8")
    (complex_dir / "complex_task.yaml").write_text("query: complex", encoding="utf-8")
    
    # Test loading simple only
    simple_tasks = load_eval_tasks(evals_dir=str(evals_dir), case_type="simple")
    assert "simple_task" in simple_tasks
    assert "complex_task" not in simple_tasks
    
    # Test loading complex only
    complex_tasks = load_eval_tasks(evals_dir=str(evals_dir), case_type="complex")
    assert "complex_task" in complex_tasks
    assert "simple_task" not in complex_tasks
    
    # Test loading all
    all_tasks = load_eval_tasks(evals_dir=str(evals_dir), case_type="all")
    assert "simple_task" in all_tasks
    assert "complex_task" in all_tasks
