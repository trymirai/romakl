# ruff: noqa: I001
from ..model import Model  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252
from dataclasses import dataclass, replace
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from lalamo import LanguageModel
from lalamo.models.language_model import GenerationConfig
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

    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]:
        result = []
        for index, prompt in enumerate(prompts):
            tokens = (
                self.model.generate_tokens(
                    jnp.asarray([prompt], dtype=jnp.int32),
                    generation_config,
                    max_output_length=token_limit,
                    keychain=Keychain.init(seed + index, sharding_config=self.model.sharding_config),
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
        result = []
        for rollout in rollouts:
            logits = self.model.decoder(
                jnp.asarray([rollout.tokens[:-1]], dtype=jnp.int32),
                jnp.arange(len(rollout.tokens) - 1, dtype=jnp.int32)[None, :],
                return_suffix_tokens=sum(rollout.mask),
                forward_pass_config=config,
                keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
            ).logits[0]
            result.append(np.asarray(logits, dtype=np.float32))
        return result
