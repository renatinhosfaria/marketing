from pathlib import Path

DOCS = [
    "docs/README.md",
    "docs/overview.md",
    "docs/architecture.md",
    "docs/backend.md",
    "docs/frontend.md",
    "docs/apis.md",
    "docs/ml.md",
    "docs/agent.md",
    "docs/facebook-ads.md",
    "docs/infra-deploy.md",
    "docs/configuration.md",
    "docs/observability.md",
    "docs/runbooks.md",
    "docs/testing.md",
    "docs/contributing.md",
]


def test_docs_exist_and_have_title():
    for rel in DOCS:
        path = Path(rel)
        assert path.exists(), f"missing {rel}"
        content = path.read_text(encoding="utf-8")
        assert content.strip().startswith("# "), f"missing H1 in {rel}"


def test_readme_sections_present():
    content = Path("docs/README.md").read_text(encoding="utf-8")
    assert "Comece aqui" in content
    assert "Mapa dos servicos" in content
    assert "Documentos" in content
