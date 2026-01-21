"""Testes para collect_results node."""
import os
import importlib.util
from datetime import datetime, timedelta

import pytest


# Carregar o modulo diretamente para evitar dependencias de __init__.py
def load_module_direct(module_name: str, relative_path: str):
    """Carrega modulo diretamente pelo path."""
    module_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', '..',
        relative_path
    )
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConvertSubagentToResult:
    """Testes para a funcao convert_subagent_to_result."""

    def test_convert_successful_result(self):
        """SubagentState com result deve retornar AgentResult com success=True."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        now = datetime.utcnow()
        subagent_state = {
            "messages": [],
            "task": {
                "description": "Analyze campaigns",
                "context": {},
                "priority": 1
            },
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {"analysis": "test data", "confidence": 0.95},
            "error": None,
            "tool_calls_count": 3,
            "started_at": now - timedelta(seconds=2),
            "completed_at": now
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "classification"
        )

        assert result["agent_name"] == "classification"
        assert result["success"] is True
        assert result["data"] == {"analysis": "test data", "confidence": 0.95}
        assert result["error"] is None
        assert len(result["tool_calls"]) == 3

    def test_convert_error_result(self):
        """SubagentState com error deve retornar AgentResult com success=False."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        now = datetime.utcnow()
        subagent_state = {
            "messages": [],
            "task": {
                "description": "Analyze anomalies",
                "context": {},
                "priority": 1
            },
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": None,
            "error": "API timeout",
            "tool_calls_count": 1,
            "started_at": now - timedelta(seconds=5),
            "completed_at": now
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "anomaly"
        )

        assert result["agent_name"] == "anomaly"
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] == "API timeout"

    def test_convert_calculates_duration(self):
        """Verifica calculo correto de duration_ms a partir dos timestamps."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        started = datetime(2024, 1, 1, 12, 0, 0, 0)
        completed = datetime(2024, 1, 1, 12, 0, 2, 500000)  # 2.5 segundos depois

        subagent_state = {
            "messages": [],
            "task": {"description": "Test", "context": {}, "priority": 1},
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {"data": "ok"},
            "error": None,
            "tool_calls_count": 0,
            "started_at": started,
            "completed_at": completed
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "forecast"
        )

        # 2.5 segundos = 2500 ms
        assert result["duration_ms"] == 2500

    def test_convert_handles_none_timestamps(self):
        """Deve retornar 0ms quando timestamps sao None."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        subagent_state = {
            "messages": [],
            "task": {"description": "Test", "context": {}, "priority": 1},
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {"data": "ok"},
            "error": None,
            "tool_calls_count": 0,
            "started_at": None,
            "completed_at": None
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "recommendation"
        )

        assert result["duration_ms"] == 0

    def test_convert_handles_only_started_at(self):
        """Deve retornar 0ms quando apenas started_at esta presente."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        subagent_state = {
            "messages": [],
            "task": {"description": "Test", "context": {}, "priority": 1},
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {"data": "ok"},
            "error": None,
            "tool_calls_count": 2,
            "started_at": datetime.utcnow(),
            "completed_at": None
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "campaign"
        )

        assert result["duration_ms"] == 0

    def test_convert_extracts_tool_calls_count(self):
        """tool_calls deve ser lista com tamanho igual a tool_calls_count."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        subagent_state = {
            "messages": [],
            "task": {"description": "Test", "context": {}, "priority": 1},
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {"data": "ok"},
            "error": None,
            "tool_calls_count": 5,
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow()
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "analysis"
        )

        assert isinstance(result["tool_calls"], list)
        assert len(result["tool_calls"]) == 5

    def test_convert_handles_empty_result(self):
        """Deve tratar result vazio (dict vazio) corretamente."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )

        subagent_state = {
            "messages": [],
            "task": {"description": "Test", "context": {}, "priority": 1},
            "config_id": 1,
            "user_id": 1,
            "thread_id": "test",
            "result": {},
            "error": None,
            "tool_calls_count": 0,
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow()
        }

        result = collect_module.convert_subagent_to_result(
            subagent_state, "classification"
        )

        # Resultado vazio ainda e sucesso (nao e None)
        assert result["success"] is True
        assert result["data"] == {}


class TestCollectResults:
    """Testes para a funcao collect_results."""

    def test_collect_results_empty_input(self):
        """Sem subagent states deve retornar dict com agent_results vazio."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        result = collect_module.collect_results(state, subagent_states=None)

        assert "agent_results" in result
        assert result["agent_results"] == {}

    def test_collect_results_empty_list(self):
        """Lista vazia de subagent states deve retornar agent_results vazio."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        result = collect_module.collect_results(state, subagent_states=[])

        assert "agent_results" in result
        assert result["agent_results"] == {}

    def test_collect_results_single_result(self):
        """Deve coletar corretamente um unico resultado de subagente."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "classification",
                "messages": [],
                "task": {"description": "Test", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"category": "performance"},
                "error": None,
                "tool_calls_count": 2,
                "started_at": now - timedelta(seconds=1),
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        assert "agent_results" in result
        assert "classification" in result["agent_results"]
        assert result["agent_results"]["classification"]["success"] is True
        assert result["agent_results"]["classification"]["data"] == {"category": "performance"}

    def test_collect_results_multiple_results(self):
        """Deve coletar corretamente multiplos resultados de subagentes."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "classification",
                "messages": [],
                "task": {"description": "Classify", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"category": "performance"},
                "error": None,
                "tool_calls_count": 2,
                "started_at": now - timedelta(seconds=1),
                "completed_at": now
            },
            {
                "agent_name": "anomaly",
                "messages": [],
                "task": {"description": "Detect anomalies", "context": {}, "priority": 2},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"anomalies": ["high_cpc"]},
                "error": None,
                "tool_calls_count": 3,
                "started_at": now - timedelta(seconds=2),
                "completed_at": now
            },
            {
                "agent_name": "forecast",
                "messages": [],
                "task": {"description": "Forecast", "context": {}, "priority": 3},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"prediction": 1500},
                "error": None,
                "tool_calls_count": 1,
                "started_at": now - timedelta(seconds=3),
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        assert len(result["agent_results"]) == 3
        assert "classification" in result["agent_results"]
        assert "anomaly" in result["agent_results"]
        assert "forecast" in result["agent_results"]

    def test_collect_results_mixed_success_error(self):
        """Deve coletar corretamente resultados mistos (sucesso e erro)."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "classification",
                "messages": [],
                "task": {"description": "Classify", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"category": "performance"},
                "error": None,
                "tool_calls_count": 2,
                "started_at": now - timedelta(seconds=1),
                "completed_at": now
            },
            {
                "agent_name": "anomaly",
                "messages": [],
                "task": {"description": "Detect anomalies", "context": {}, "priority": 2},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": None,
                "error": "API Error: Rate limited",
                "tool_calls_count": 1,
                "started_at": now - timedelta(seconds=2),
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        assert len(result["agent_results"]) == 2
        assert result["agent_results"]["classification"]["success"] is True
        assert result["agent_results"]["anomaly"]["success"] is False
        assert result["agent_results"]["anomaly"]["error"] == "API Error: Rate limited"

    def test_collect_results_preserves_existing(self):
        """Deve mesclar com agent_results existentes no estado."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        # Adicionar resultado existente
        state["agent_results"] = {
            "recommendation": {
                "agent_name": "recommendation",
                "success": True,
                "data": {"recommendations": ["optimize budget"]},
                "error": None,
                "duration_ms": 1500,
                "tool_calls": ["tool_1"]
            }
        }

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "classification",
                "messages": [],
                "task": {"description": "Classify", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"category": "performance"},
                "error": None,
                "tool_calls_count": 2,
                "started_at": now - timedelta(seconds=1),
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        # Deve ter ambos os resultados
        assert len(result["agent_results"]) == 2
        assert "recommendation" in result["agent_results"]
        assert "classification" in result["agent_results"]
        # Resultado existente deve estar preservado
        assert result["agent_results"]["recommendation"]["data"] == {"recommendations": ["optimize budget"]}

    def test_collect_results_handles_missing_fields(self):
        """Deve tratar graciosamente estados incompletos."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        # Estado incompleto - sem alguns campos
        subagent_states = [
            {
                "agent_name": "classification",
                "result": {"data": "partial"},
                # Campos faltando: messages, task, tool_calls_count, timestamps, etc.
            }
        ]

        # Nao deve lancar excecao
        result = collect_module.collect_results(state, subagent_states=subagent_states)

        assert "agent_results" in result
        assert "classification" in result["agent_results"]
        assert result["agent_results"]["classification"]["success"] is True
        assert result["agent_results"]["classification"]["data"] == {"data": "partial"}


class TestCollectResultsEdgeCases:
    """Testes para casos extremos do collect_results."""

    def test_collect_results_overwrite_existing(self):
        """Novo resultado deve sobrescrever resultado existente do mesmo agente."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )
        # Adicionar resultado existente
        state["agent_results"] = {
            "classification": {
                "agent_name": "classification",
                "success": True,
                "data": {"old": "data"},
                "error": None,
                "duration_ms": 1000,
                "tool_calls": []
            }
        }

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "classification",
                "messages": [],
                "task": {"description": "Reclassify", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"new": "data"},
                "error": None,
                "tool_calls_count": 3,
                "started_at": now - timedelta(seconds=2),
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        # Resultado deve ser sobrescrito
        assert result["agent_results"]["classification"]["data"] == {"new": "data"}

    def test_collect_results_with_zero_tool_calls(self):
        """Deve funcionar com zero tool_calls."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        now = datetime.utcnow()
        subagent_states = [
            {
                "agent_name": "forecast",
                "messages": [],
                "task": {"description": "Test", "context": {}, "priority": 1},
                "config_id": 1,
                "user_id": 1,
                "thread_id": "test",
                "result": {"prediction": 100},
                "error": None,
                "tool_calls_count": 0,
                "started_at": now,
                "completed_at": now
            }
        ]

        result = collect_module.collect_results(state, subagent_states=subagent_states)

        assert result["agent_results"]["forecast"]["tool_calls"] == []

    def test_collect_results_returns_dict_format(self):
        """Deve retornar dict no formato esperado para atualizacao de estado."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        state_module = load_module_direct(
            "state",
            "app/agent/orchestrator/state.py"
        )

        state = state_module.create_initial_orchestrator_state(
            config_id=1, user_id=1, thread_id="test"
        )

        result = collect_module.collect_results(state, subagent_states=None)

        # Deve retornar dict com chave 'agent_results'
        assert isinstance(result, dict)
        assert "agent_results" in result
        assert isinstance(result["agent_results"], dict)


class TestConvertSubagentImport:
    """Testes de importacao do modulo collect_results."""

    def test_collect_results_import(self):
        """collect_results deve ser importavel."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        assert collect_module.collect_results is not None

    def test_convert_subagent_to_result_import(self):
        """convert_subagent_to_result deve ser importavel."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        assert collect_module.convert_subagent_to_result is not None

    def test_collect_results_is_callable(self):
        """collect_results deve ser uma funcao."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        assert callable(collect_module.collect_results)

    def test_convert_subagent_to_result_is_callable(self):
        """convert_subagent_to_result deve ser uma funcao."""
        collect_module = load_module_direct(
            "collect_results",
            "app/agent/orchestrator/nodes/collect_results.py"
        )
        assert callable(collect_module.convert_subagent_to_result)
