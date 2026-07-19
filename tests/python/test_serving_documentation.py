from pathlib import Path


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
