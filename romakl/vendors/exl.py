from dataclasses import dataclass

import numpy as np

from ..model import Model, StudentModel  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class ExLlamaModel(Model, StudentModel):
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        import torch
        from exllamav3 import Config
        from exllamav3 import Model as ExLlamaV3Model

        model = ExLlamaV3Model.from_config(Config.from_directory(str(self.path)))
        model.load(device="cuda:0")
        result = []
        for rollout in rollouts:
            inputs = torch.tensor([rollout.tokens[:-1]], dtype=torch.long)
            logits = model.forward(inputs, {"last_tokens_only": sum(rollout.mask)})[0]
            result.append(logits.half().cpu().numpy())
        model.unload()
        torch.cuda.empty_cache()
        return result
