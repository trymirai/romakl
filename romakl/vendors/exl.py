from dataclasses import dataclass

import numpy as np

from ..model import GenerationParams, Model  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class ExLlamaModel(Model):
    def generate(self, prompts: list[list[int]], generation_params: GenerationParams) -> list[Rollout]:
        raise NotImplementedError("ExLlama generation is not supported")

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        import torch
        from exllamav3 import Config
        from exllamav3 import Model as ExLlamaV3Model

        model = ExLlamaV3Model.from_config(Config.from_directory(str(self.path)))
        model.load(
            max_chunk_size=max(len(rollout.tokens) - 1 for rollout in rollouts),
            max_output_size=max(sum(rollout.mask) for rollout in rollouts),
            max_output_factor=3,
            max_batch_size=1,
        )
        result = []
        for rollout in rollouts:
            inputs = torch.tensor([rollout.tokens[:-1]], dtype=torch.long)
            logits = model.forward(inputs, {"last_tokens_only": sum(rollout.mask)})
            result.append(logits[0].float().cpu().numpy())
        model.unload()
        torch.cuda.empty_cache()
        return result
