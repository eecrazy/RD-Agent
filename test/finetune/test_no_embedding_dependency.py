import os
from unittest.mock import Mock, patch

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune import _effective_api_max_workers
from rdagent.components.coder.finetune.conf import (
    DEFAULT_RESPONSES_API_MAX_WORKERS,
    DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT,
    effective_data_processing_timeout,
    get_data_processing_env,
)
from rdagent.core.scenario import Scenario
from rdagent.scenarios.finetune.train.runner import LLMFinetuneRunner


def test_responses_adapter_uses_bounded_data_generation_workers() -> None:
    with patch.dict(os.environ, {"FT_API_PROTOCOL": "responses"}, clear=True):
        assert _effective_api_max_workers() == DEFAULT_RESPONSES_API_MAX_WORKERS

    with patch.dict(
        os.environ,
        {
            "FT_API_PROTOCOL": "responses",
            "FT_RESPONSES_API_MAX_WORKERS": "1",
        },
        clear=True,
    ):
        assert _effective_api_max_workers() == 1

    with patch.dict(os.environ, {"FT_API_PROTOCOL": "chat_completions"}, clear=True):
        assert _effective_api_max_workers() == FT_RD_SETTING.api_max_workers


def test_data_processing_environment_uses_effective_api_worker_limit() -> None:
    environment = Mock()
    with (
        patch("rdagent.components.coder.finetune.conf.get_ft_env", return_value=environment),
        patch.dict(os.environ, {"FT_API_PROTOCOL": "responses"}, clear=True),
    ):
        returned_environment, env_vars = get_data_processing_env(is_debug=True)

    assert returned_environment is environment
    assert env_vars["FT_API_MAX_WORKERS"] == str(DEFAULT_RESPONSES_API_MAX_WORKERS)
    assert env_vars["CUDA_VISIBLE_DEVICES"] == ""

    with (
        patch("rdagent.components.coder.finetune.conf.get_ft_env", return_value=environment),
        patch.dict(os.environ, {"FT_API_PROTOCOL": "chat_completions"}, clear=True),
    ):
        _, env_vars = get_data_processing_env(is_debug=False)

    assert env_vars["FT_API_MAX_WORKERS"] == str(FT_RD_SETTING.api_max_workers)


def test_responses_reproduction_uses_extended_full_data_timeout() -> None:
    configured_timeout = 6 * 60 * 60
    with patch.dict(
        os.environ,
        {
            "FT_API_PROTOCOL": "responses",
            "FT_EXPERIMENT_ID": "main/aime25/run-2",
        },
        clear=True,
    ):
        assert effective_data_processing_timeout() == DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT

    with patch.dict(
        os.environ,
        {
            "FT_API_PROTOCOL": "responses",
            "FT_EXPERIMENT_ID": "main/aime25/run-2",
            "FT_RESPONSES_DATA_PROCESSING_TIMEOUT": str(configured_timeout),
        },
        clear=True,
    ):
        assert effective_data_processing_timeout() == configured_timeout

    with patch.dict(os.environ, {"FT_API_PROTOCOL": "responses"}, clear=True):
        assert effective_data_processing_timeout() == FT_RD_SETTING.data_processing_timeout


def test_finetune_runner_disables_optional_knowledge_generation() -> None:
    runner = LLMFinetuneRunner(Mock(spec=Scenario))

    assert runner.knowledge_self_gen is False
    # The queried-knowledge object is still made available to the evolving
    # strategy, so disabling persistence does not alter its feedback loop.
    assert runner.with_knowledge is True
