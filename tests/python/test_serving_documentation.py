from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]


def test_s5_documentation_consistency() -> None:
    architecture = (ROOT / "docs/serving/architecture.md").read_text()
    kv_cache = (ROOT / "docs/serving/kv_cache.md").read_text()
    serving_docs = "\n".join(
        path.read_text() for path in (ROOT / "docs/serving").glob("*.md")
    )

    assert "Phase S5 implements metadata-only prefix caching" in architecture
    assert "(last_access_epoch, physical_block_id)" in architecture
    assert "both `InUse` and `Cached` blocks" in kv_cache
    assert "Preemption, swapping, distributed caching" in architecture
    assert "remain future architecture" in architecture
    removed_metric = "eviction_failure" + "_count"
    assert removed_metric not in serving_docs


def test_s6_documentation_and_github_presentation() -> None:
    final_report_path = ROOT / "docs/serving/final_report.md"
    assert final_report_path.is_file()
    final_report = final_report_path.read_text()
    readme = (ROOT / "README.md").read_text()
    development_plan = (ROOT / "docs/serving/development_plan.md").read_text()
    architecture = (ROOT / "docs/serving/architecture.md").read_text()
    comparison = (ROOT / "docs/serving/engine_comparison.md").read_text()

    assert "docs/serving/final_report.md" in readme
    assert "Mini LLM Serving Engine" in readme
    assert "**Status: implemented.**" in development_plan
    assert all("SIMULATED" in row for row in every_serving_result_table(final_report))
    assert "prefix cache future work" not in (architecture + final_report).lower()
    assert "(last_access_epoch, physical_block_id)" in architecture
    assert "Preemption" in architecture and "future architecture" in architecture
    assert "not a compatibility, performance, or feature-parity claim" in comparison
    assert (ROOT / "docs/serving/workload_schema.md").is_file()
    assert "serving/workload_schema.md" in (ROOT / "docs/README.md").read_text()
    assert "results/serving/raw/*.jsonl" not in final_report
    assert "--command-manifest" in final_report
    assert "--update-reference" in final_report
    extended = json.loads((ROOT / "configs/serving/matrix_extended.json").read_text())
    assert extended["max_default_runs"] == len(extended["configs"]) == 24
    for relative in (
        "benchmarks/run_serving_matrix.py",
        "benchmarks/analyze_serving_results.py",
        "configs/serving/matrix_small.json",
        "scripts/run_serving_demo.sh",
        "scripts/verify_serving_project.sh",
    ):
        assert (ROOT / relative).exists(), relative


def every_serving_result_table(document: str) -> list[str]:
    """Return joined data rows from report tables that contain an Evidence column."""
    lines = document.splitlines()
    rows: list[str] = []
    in_result_table = False
    for line in lines:
        if line.startswith("|") and "Evidence" in line:
            in_result_table = True
            continue
        if in_result_table and line.startswith("|"):
            if "---" not in line:
                rows.append(line)
        elif in_result_table:
            in_result_table = False
    assert rows, "final report must contain serving result tables"
    return rows
