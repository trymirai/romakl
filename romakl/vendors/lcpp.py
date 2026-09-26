from contextlib import closing
from dataclasses import dataclass
from itertools import islice

import numpy as np
from lalamo.models.language_model import GenerationConfig
from llama_cpp import Llama

from ..model import Model  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252


@dataclass(frozen=True)
class GGUFModel(Model):
    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]:
        p = generation_config
        if p.banned_tokens or p.suffix_repetition_length:
            raise NotImplementedError("llama.cpp generation does not support banned tokens or suffix repetition")
        result = []
        with closing(
            Llama(
                str(self.path),
                n_ctx=max(len(prompt) + token_limit for prompt in prompts),
                n_gpu_layers=-1,
                verbose=False,
            )
        ) as model:
            for index, prompt in enumerate(prompts):
                model.set_seed(seed + index)
                stream = model.generate(
                    prompt,
                    temp=1.0 if p.temperature is None else p.temperature,
                    top_k=0 if p.top_k is None else p.top_k,
                    top_p=1.0 if p.top_p is None else p.top_p,
                    min_p=0.0 if p.min_p is None else p.min_p,
                    repeat_penalty=1.0 if p.repetition_penalty is None else p.repetition_penalty,
                    presence_penalty=0.0 if p.presence_penalty is None else p.presence_penalty,
                    frequency_penalty=0.0 if p.frequency_penalty is None else p.frequency_penalty,
                )
                tokens = []
                for token in islice(stream, token_limit):
                    tokens.append(token)
                    # EOT is exposed only by the underlying llama.cpp model.
                    if token in (*p.stop_token_ids, model.token_eos(), model._model.token_eot()):  # noqa: SLF001
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
                result.append(model.scores[model.n_tokens - sum(rollout.mask) : model.n_tokens].astype(np.float16))
        return result
