"""LLM Fine-tuning Memory Constraints Calculator

Calculate max supported seq_len for each fine-tuning method.
Based on EleutherAI Transformer Math: https://blog.eleuther.ai/transformer-math/
"""

import re
from collections.abc import Sequence


class MemoryEstimator:
    """Calculate memory constraints for fine-tuning methods."""

    # Memory factors (GB per billion parameters)
    MEM_FACTOR = {
        "full": 18,  # bf16 params + bf16 grads + fp32 optimizer states
        "base_bf16": 2,  # bf16 params only (frozen)
        "base_4bit": 0.5,  # 4-bit quantized params
        "trainable": 18,  # trainable params
    }

    # Architecture estimation: params_b -> (hidden_dim, num_layers)
    ARCH = {
        3: (2048, 24),
        7: (4096, 32),
        13: (5120, 40),
        34: (6144, 48),
        70: (8192, 80),
    }

    DEFAULT_LORA_RANK = 64

    def __init__(
        self,
        params_b: float,
        gpu_mem: float,
        num_gpus: int,
        max_position_embeddings: int = 32768,
        gpu_name: str = "GPU",
        resource_scope: str = "runtime-visible physical resources",
    ):
        self.params_b = params_b
        self.gpu_mem = gpu_mem
        self.num_gpus = num_gpus
        self.total_mem = gpu_mem * num_gpus
        self.max_ctx = max_position_embeddings
        self.gpu_name = gpu_name
        self.resource_scope = resource_scope

        # Estimate architecture
        self.hidden, self.layers = next(
            (v for k, v in self.ARCH.items() if params_b <= k),
            (8192, 96),
        )

    @classmethod
    def from_model_name(
        cls,
        name: str,
        gpu_mem: float,
        num_gpus: int,
        model_specs: str = "",
        gpu_name: str = "GPU",
        resource_scope: str = "runtime-visible physical resources",
    ) -> "MemoryEstimator":
        """Create from model name and specs."""
        # Parse params from name: Qwen2.5-7B -> 7.0
        match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        params_b = float(match.group(1)) if match else 7.0

        # Parse max_position_embeddings from specs
        max_ctx = 32768
        if model_specs:
            ctx_match = re.search(r"max_position_embeddings:\s*(\d+)", model_specs)
            if ctx_match:
                max_ctx = int(ctx_match.group(1))

        return cls(
            params_b,
            gpu_mem,
            num_gpus,
            max_ctx,
            gpu_name=gpu_name,
            resource_scope=resource_scope,
        )

    def _base_memory(self, method: str) -> float:
        """Base memory without activations (GB)."""
        lora_p = 2 * self.DEFAULT_LORA_RANK * self.hidden * 4 * self.layers / 1e9

        if method == "full" or method == "full_gc":
            return self.params_b * self.MEM_FACTOR["full"]
        if method == "lora":
            return self.params_b * self.MEM_FACTOR["base_bf16"] + lora_p * self.MEM_FACTOR["trainable"]
        if method == "qlora":
            return self.params_b * self.MEM_FACTOR["base_4bit"] + lora_p * self.MEM_FACTOR["trainable"]
        return 0

    def _activation_factor(self, method: str) -> float:
        """Activation memory factor (gradient checkpointing reduces this)."""
        return 0.35 if method == "full_gc" else 1.0

    def _find_max_seq_len(self, method: str, batch_size: int = 1) -> int:
        """Find max seq_len that fits in memory."""
        available = self.total_mem * 0.9
        base = self._base_memory(method)
        remaining = available - base * 1.2

        if remaining <= 0:
            return 0

        act_factor = self._activation_factor(method)
        # activation = seq * hidden * layers * 8 * batch / 1e9 * act_factor * 1.2
        max_seq = int(remaining * 1e9 / (self.hidden * self.layers * 8 * batch_size * act_factor * 1.2))
        return max_seq  # Don't cap at max_ctx here, show raw capability

    def estimate(self, methods: Sequence[str] | None = None) -> dict[str, int]:
        """Calculate max seq_len for each method (batch=1)."""
        selected_methods = methods if methods is not None else ("full", "full_gc", "lora", "qlora")
        return {method: self._find_max_seq_len(method) for method in selected_methods}

    def format(
        self,
        estimates: dict[str, int] | None = None,
        methods: Sequence[str] | None = None,
    ) -> str:
        """Format as constraint table."""
        if estimates is None:
            estimates = self.estimate(methods)

        lines = [
            "## Training Resource Memory Constraints",
            f"**Resource scope**: {self.resource_scope}",
            (
                f"**Hardware**: {self.num_gpus}x {self.gpu_name}, "
                f"{self.gpu_mem:.0f}GB each = {self.total_mem:.0f}GB total"
            ),
            f"**Model**: {self.params_b}B parameters",
            f"**Model max_position_embeddings**: {self.max_ctx}",
            "",
            "| Method | Max seq_len (batch=1) |",
            "|--------|----------------------|",
        ]

        for method, max_seq in estimates.items():
            if max_seq > 0:
                lines.append(f"| {method} | {max_seq} |")
            else:
                lines.append(f"| {method} | Not viable |")

        lines.append("")
        lines.append("**Note**: Choose `cutoff_len` <= min(max_seq_len, max_position_embeddings)")
        lines.append("- Larger `cutoff_len` enables longer CoT but reduces batch_size")
        visible_methods = set(estimates)
        if "qlora" in visible_methods:
            lines.append("- Method quality: full > lora > qlora (when all can support your seq_len needs)")
        elif {"full", "lora"} & visible_methods:
            lines.append("- Compare only methods allowed by the active training policy")

        return "\n".join(lines)
