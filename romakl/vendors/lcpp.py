from contextlib import closing
from dataclasses import dataclass

import numpy as np
from llama_cpp import Llama

from ..model import Model, StudentModel  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class GGUFModel(Model, StudentModel):
    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        result = []
        with closing(
            Llama(
                str(self.path),
                n_ctx=max(len(rollout.tokens) - 1 for rollout in rollouts),
                n_gpu_layers=-1,
                logits_all=True,
                verbose=False,
            )
        ) as model:
            for rollout in rollouts:
                model.reset()
                model.eval(rollout.tokens[:-1])
                result.append(model.scores[model.n_tokens - sum(rollout.mask) : model.n_tokens].astype(np.float16))
        return result
