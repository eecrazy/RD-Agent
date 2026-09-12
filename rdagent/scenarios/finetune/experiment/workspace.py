"""FT-specific workspace with lightweight code and durable model checkpoints."""

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rdagent.components.coder.finetune.conf import FT_YAML_FILE_NAME
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    formal_training_provenance_files,
    validate_recorded_formal_training_evidence,
)
from rdagent.utils.env import CacheKeyFunc, DockerEnv, LocalEnv

if TYPE_CHECKING:
    from rdagent.utils.env import Env

from rdagent.utils.env import EnvResult


class FTWorkspace(FBWorkspace):
    """
    Fine-tuning workspace with durable inference checkpoints and unified Docker logging.

    The generic in-memory workspace checkpoint intentionally excludes large
    files.  Fine-tuning outputs are different from disposable datasets: the
    validation-selected adapter must remain available for the one-shot final
    test.  We therefore keep the small code checkpoint in memory and preserve
    the top-level inference artifact and each saved trainer checkpoint from
    ``output/`` in a sibling store.  Intermediate weights are needed for strict
    validation-only model selection after the search has finished.  Optimizer
    and RNG state are deliberately omitted because they are not needed for
    inference.  Hard links avoid duplicating model bytes on the normal
    same-filesystem path.  Formal training inputs/evidence are separately
    copied byte-for-byte (never hard-linked), so generated datasets remain
    auditable after the lightweight workspace archive drops large files.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Configure checkpoint to save essential files for training
        # Large inference artifacts are handled by _save_model_checkpoint below.
        RD_AGENT_SETTINGS.workspace_ckp_white_list_names = [
            FT_YAML_FILE_NAME,  # train.yaml - training config
            "dataset_info.json",  # LlamaFactory dataset config
        ]
        RD_AGENT_SETTINGS.workspace_ckp_size_limit = 100 * 1024

        self._model_checkpoint_path: Path | None = None
        self._formal_training_checkpoint_path: Path | None = None

    @property
    def _durable_model_checkpoint_path(self) -> Path:
        return self.workspace_path.parent / ".ft_model_checkpoints" / self.workspace_path.name / "output"

    @property
    def _durable_formal_training_path(self) -> Path:
        return self.workspace_path.parent / ".ft_model_checkpoints" / self.workspace_path.name / "formal_training"

    @staticmethod
    def _link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            return shutil.copy2(source, destination)
        else:
            return destination

    @staticmethod
    def _has_model_weights(output_path: Path) -> bool:
        patterns = (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
        )
        return any(any(output_path.glob(pattern)) for pattern in patterns)

    def _save_model_checkpoint(self) -> None:
        source = self.workspace_path / "output"
        if not source.is_dir() or not self._has_model_weights(source):
            # A CoSTEER candidate is derived in place and can reuse a workspace
            # id that previously contained a trained model.  Do not let that
            # older durable output make this code-only candidate look trained.
            shutil.rmtree(self._durable_model_checkpoint_path, ignore_errors=True)
            self._model_checkpoint_path = None
            return

        destination = self._durable_model_checkpoint_path
        temporary = destination.with_name("output.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        for item in source.iterdir():
            if item.is_file():
                self._link_or_copy(str(item), str(temporary / item.name))
            elif item.is_symlink():
                (temporary / item.name).symlink_to(item.readlink())
            elif item.is_dir() and item.name.startswith("checkpoint-") and self._has_model_weights(item):
                checkpoint_destination = temporary / item.name
                checkpoint_destination.mkdir()
                for checkpoint_item in item.iterdir():
                    if checkpoint_item.is_symlink():
                        (checkpoint_destination / checkpoint_item.name).symlink_to(checkpoint_item.readlink())
                    elif checkpoint_item.is_file() and checkpoint_item.name not in {
                        "optimizer.pt",
                        "scheduler.pt",
                        "rng_state.pth",
                    }:
                        self._link_or_copy(
                            str(checkpoint_item),
                            str(checkpoint_destination / checkpoint_item.name),
                        )
        if not self._has_model_weights(temporary):
            shutil.rmtree(temporary, ignore_errors=True)
            message = f"Unable to preserve fine-tuned model weights from {source}"
            raise RuntimeError(message)
        shutil.rmtree(destination, ignore_errors=True)
        temporary.replace(destination)
        self._model_checkpoint_path = destination

    def _save_formal_training_checkpoint(self) -> None:
        evidence = self.workspace_path / FORMAL_TRAINING_EVIDENCE_FILE
        destination = self._durable_formal_training_path
        if not evidence.is_file():
            shutil.rmtree(destination, ignore_errors=True)
            self._formal_training_checkpoint_path = None
            return
        if self._model_checkpoint_path is None:
            # Small files survive the generic workspace copy while model
            # outputs intentionally do not.  A newly derived code-only
            # candidate can therefore inherit its parent's evidence without
            # inheriting the model that evidence authenticates.  Such evidence
            # is stale by construction and must not enter either checkpoint.
            evidence.unlink()
            shutil.rmtree(destination, ignore_errors=True)
            self._formal_training_checkpoint_path = None
            durable_root = destination.parent
            try:
                durable_root.rmdir()
            except OSError:
                pass
            return

        relative_files = formal_training_provenance_files(self.workspace_path)
        temporary = destination.with_name("formal_training.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            for relative in relative_files:
                source = self.workspace_path / relative
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # A hard link would allow later workspace mutations to rewrite
                # historical proof, so provenance always uses a real copy.
                shutil.copy2(source, target, follow_symlinks=True)
                if source.read_bytes() != target.read_bytes():
                    raise RuntimeError(f"Formal provenance copy changed bytes: {source}")
            validate_recorded_formal_training_evidence(
                temporary,
                output_path=self._model_checkpoint_path,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        shutil.rmtree(destination, ignore_errors=True)
        temporary.replace(destination)
        self._formal_training_checkpoint_path = destination

    def create_ws_ckp(self) -> None:
        """Checkpoint code in memory and selected inference weights on disk."""
        # Resolve heavyweight state first.  In particular, stale formal
        # evidence must be removed before the generic zip captures small files.
        self._save_model_checkpoint()
        self._save_formal_training_checkpoint()
        super().create_ws_ckp()

    def recover_ws_ckp(self) -> None:
        """Restore both the lightweight workspace and its selected model."""
        model_checkpoint = self._model_checkpoint_path
        formal_checkpoint = self._formal_training_checkpoint_path
        super().recover_ws_ckp()
        if formal_checkpoint is not None:
            if not formal_checkpoint.is_dir():
                raise RuntimeError(f"Formal training checkpoint is missing: {formal_checkpoint}")
            shutil.copytree(
                formal_checkpoint,
                self.workspace_path,
                copy_function=shutil.copy2,
                symlinks=False,
                dirs_exist_ok=True,
            )
        if model_checkpoint is None:
            if formal_checkpoint is not None:
                raise RuntimeError("Formal training checkpoint exists without a model checkpoint")
            return
        if not model_checkpoint.is_dir() or not self._has_model_weights(model_checkpoint):
            message = f"Fine-tuned model checkpoint is missing: {model_checkpoint}"
            raise RuntimeError(message)
        output = self.workspace_path / "output"
        shutil.copytree(
            model_checkpoint,
            output,
            copy_function=self._link_or_copy,
            symlinks=True,
            dirs_exist_ok=True,
        )
        if formal_checkpoint is not None:
            validate_recorded_formal_training_evidence(
                self.workspace_path,
                output_path=output,
            )

    def run(
        self,
        env: "Env",
        entry: str,
        env_vars: dict | None = None,
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> "EnvResult":
        """Execute the code in the environment with unified Docker logging.

        Args:
            env: The environment to run in (DockerEnv, LocalEnv, etc.)
            entry: The command to execute
            env_vars: Optional additional environment variables (e.g., LLM API keys)
                     Will be merged with default {"PYTHONPATH": "./"}
            cache_key_extra_func: Optional extra function for cache key calculation
            cache_files_to_extract: Optional list of files to extract from cache

        Returns:
            EnvResult with stdout, exit_code, running_time
        """
        self.prepare()
        self.inject_files(**self.file_dict)

        # Merge default env with custom env_vars
        run_env = {"PYTHONPATH": "./"}
        if env_vars:
            run_env.update(env_vars)

        result = env.run(
            entry,
            str(self.workspace_path),
            env=run_env,
            cache_key_extra_func=cache_key_extra_func,
            cache_files_to_extract=cache_files_to_extract,
        )

        # Unified execution logging for FT scenario (supports both Docker and Conda)
        if isinstance(env, DockerEnv):
            tag_prefix = "docker_run"
        elif isinstance(env, LocalEnv):
            tag_prefix = "conda_run"
        else:
            tag_prefix = "env_run"

        logger.log_object(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "running_time": result.running_time,
                "entry": entry,
                "workspace_path": str(self.workspace_path),
            },
            tag=f"{tag_prefix}.FTWorkspace",
        )

        return result
