from contextlib import closing
from dataclasses import dataclass
from itertools import islice

import numpy as np
from llama_cpp import Llama

from ..model import GenerationParams, Model  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class GGUFModel(Model):
    def generate(self, prompts: list[list[int]], generation_params: GenerationParams) -> list[Rollout]:
        p = generation_params
        result = []
        with closing(
            Llama(
                str(self.path),
                n_ctx=max(len(prompt) + p.token_limit for prompt in prompts),
                n_gpu_layers=-1,
                verbose=False,
            )
        ) as model:
            for index, prompt in enumerate(prompts):
                model.set_seed(p.seed + index)
                stream = model.generate(
                    prompt,
                    temp=p.temperature,
                    top_k=p.top_k,
                    top_p=p.top_p,
                    min_p=p.min_p,
                    repeat_penalty=p.repetition_penalty,
                    presence_penalty=p.presence_penalty,
                    frequency_penalty=p.frequency_penalty,
                )
                tokens = []
                for token in islice(stream, p.token_limit):
                    tokens.append(token)
                    # EOT is exposed only by the underlying llama.cpp model.
                    if token in (model.token_eos(), model._model.token_eot()):  # noqa: SLF001
                        break
                result.append(Rollout([*prompt, *tokens], [0] * len(prompt) + [1] * len(tokens)))
        return result

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        result = []
        with closing(
            Llama(
                str(self.path),
                n_ctx=max(len(r.tokens) - 1 for r in rollouts),
                n_gpu_layers=-1,
                logits_all=True,
                verbose=False,
            )
        ) as model:
            for rollout in rollouts:
                model.reset()
                model.eval(rollout.tokens[:-1])
                result.append(model.scores[model.n_tokens - sum(rollout.mask) : model.n_tokens].copy())
        return result
