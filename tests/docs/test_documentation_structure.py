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


def test_overview_sections_present():
    content = Path("docs/overview.md").read_text(encoding="utf-8")
    assert "Objetivo" in content
    assert "Escopo" in content
    assert "Nao objetivos" in content


def test_architecture_sections_present():
    content = Path("docs/architecture.md").read_text(encoding="utf-8")
    assert "Componentes" in content
    assert "Fluxo de dados" in content
    assert "Portas" in content


def test_backend_sections_present():
    content = Path("docs/backend.md").read_text(encoding="utf-8")
    assert "Entrypoints" in content
    assert "Rotas" in content
    assert "Celery" in content


def test_apis_sections_present():
    content = Path("docs/apis.md").read_text(encoding="utf-8")
    assert "ML" in content
    assert "Agente" in content
    assert "Facebook Ads" in content


def test_ml_sections_present():
    content = Path("docs/ml.md").read_text(encoding="utf-8")
    assert "Previsoes" in content
    assert "Classificacoes" in content
    assert "Recomendacoes" in content


def test_agent_sections_present():
    content = Path("docs/agent.md").read_text(encoding="utf-8")
    assert "Modo de operacao" in content
    assert "Streaming" in content
    assert "Multi-agent" in content


def test_facebook_ads_sections_present():
    content = Path("docs/facebook-ads.md").read_text(encoding="utf-8")
    assert "OAuth" in content
    assert "Sync" in content
    assert "Insights" in content


def test_frontend_sections_present():
    content = Path("docs/frontend.md").read_text(encoding="utf-8")
    assert "Scripts" in content
    assert "Variaveis de ambiente" in content


def test_infra_sections_present():
    content = Path("docs/infra-deploy.md").read_text(encoding="utf-8")
    assert "Docker Compose" in content
    assert "Healthchecks" in content


def test_configuration_sections_present():
    content = Path("docs/configuration.md").read_text(encoding="utf-8")
    assert "Variaveis principais" in content
    assert "Exemplo de .env" in content


def test_observability_sections_present():
    content = Path("docs/observability.md").read_text(encoding="utf-8")
    assert "Logs" in content
    assert "Trace" in content
