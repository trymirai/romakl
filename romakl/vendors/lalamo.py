# ruff: noqa: I001
from ..model import GenerationParams, Model  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252
from dataclasses import dataclass, replace
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from lalamo import LanguageModel
from lalamo.module import Keychain
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.token_mixer import AttentionImplementation
from lalamo.utils.sharding import ShardingConfig


@dataclass(frozen=True)
class LalamoModel(Model):
    model: LanguageModel

    @classmethod
    def load(cls, path: Path | str) -> "LalamoModel":
        model = LanguageModel.load(path, ShardingConfig.replicated(), dtype=jnp.bfloat16)
        return cls(model.token_codec, path, model)

    def generate(self, prompts: list[list[int]], generation_params: GenerationParams) -> list[Rollout]:
        params = generation_params
        config = replace(
            self.model.config.generation_config,
            **{k: v for k, v in vars(params).items() if k not in ("seed", "token_limit")},
        )
        result = []
        for index, prompt in enumerate(prompts):
            tokens = (
                self.model.generate_tokens(
                    jnp.asarray([prompt], dtype=jnp.int32),
                    config,
                    max_output_length=params.token_limit,
                    keychain=Keychain.init(params.seed + index, sharding_config=self.model.sharding_config),
                )
                .token_ids[0]
                .tolist()
            )
            tokens = self.model.trim_at_eos(tokens)
            result.append(Rollout([*prompt, *tokens], [0] * len(prompt) + [1] * len(tokens)))
        return result

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        # Lalamo's Pallas prefill path rejects this suffix forward pass.
        config = DecoderForwardPassConfig.for_inference()
        transformer_config = config.transformer_forward_pass_config
        mixer_config = replace(
            transformer_config.mixer_forward_pass_config,
            attention_implementation=AttentionImplementation.XLA,
        )
        config = replace(
            config,
            transformer_forward_pass_config=replace(transformer_config, mixer_forward_pass_config=mixer_config),
        )
        return [
            np.asarray(
                self.model.decoder(
                    jnp.asarray([r.tokens[:-1]], dtype=jnp.int32),
                    jnp.arange(len(r.tokens) - 1, dtype=jnp.int32)[None, :],
                    return_suffix_tokens=sum(r.mask),
                    forward_pass_config=config,
                    keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
                ).logits[0],
                dtype=np.float32,
            )
            for r in rollouts
        ]
