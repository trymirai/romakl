# ruff: noqa: I001
from ..model import Model, StudentModel, TeacherModel  # noqa: TID252
from ..rollouts import Rollout  # noqa: TID252
from dataclasses import dataclass, replace
from itertools import batched
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
class LalamoModel(Model, TeacherModel, StudentModel):
    model: LanguageModel

    @classmethod
    def load(cls, path: Path | str) -> "LalamoModel":
        model = LanguageModel.load(path, ShardingConfig.replicated(), dtype=jnp.bfloat16)
        return cls(model.token_codec, path, model)

    def generate(
        self, prompts: list[list[int]], generation_config: GenerationConfig, *, seed: int, token_limit: int
    ) -> list[Rollout]:
        result = []
        for batch in batched(prompts, 1):
            prompt_lengths = [len(prompt) for prompt in batch]
            max_prompt_length = max(prompt_lengths)
            padded_prompts = [prompt + [0] * (max_prompt_length - len(prompt)) for prompt in batch]
            generated = self.model.generate_tokens(
                jnp.asarray(padded_prompts, dtype=jnp.int32),
                generation_config,
                prompt_lengths_without_padding=jnp.asarray(prompt_lengths, dtype=jnp.int32),
                max_output_length=token_limit,
                keychain=Keychain.init(seed, sharding_config=self.model.sharding_config),
            ).token_ids.tolist()
            for prompt, generated_tokens in zip(batch, generated, strict=True):
                tokens = self.model.trim_at_eos(generated_tokens)
                result.append(Rollout([*prompt, *tokens], [0] * len(prompt) + [1] * len(tokens)))
        return result

    def logits(self, rollouts: list[Rollout]) -> list[np.ndarray]:
        # Lalamo's Pallas attention only supports one query token; scoring needs every logit.
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
        chunk_size = 64
        result = []
        for batch in batched(rollouts, 1):
            lengths = [len(rollout.tokens) - 1 for rollout in batch]
            padded_length = ((max(lengths) + chunk_size - 1) // chunk_size) * chunk_size
            token_ids = jnp.asarray(
                [
                    rollout.tokens[:-1] + [0] * (padded_length - length)
                    for rollout, length in zip(batch, lengths, strict=True)
                ],
                dtype=jnp.int32,
            )
            positions = jnp.broadcast_to(jnp.arange(padded_length, dtype=jnp.int32), token_ids.shape)
            lengths_without_padding = jnp.asarray(lengths, dtype=jnp.int32)
            state = self.model.decoder.init_static_state(
                len(batch), padded_length + 1, config.embedding_forward_pass_config.activation_dtype
            )
            logits_by_rollout: list[list[np.ndarray]] = [[] for _ in batch]
            for chunk_start in range(0, padded_length, chunk_size):
                decoded = self.model.decoder(
                    token_ids[:, chunk_start : chunk_start + chunk_size],
                    positions[:, chunk_start : chunk_start + chunk_size],
                    state=state,
                    return_updated_state=True,
                    lengths_without_padding=jnp.clip(lengths_without_padding - chunk_start, 0, chunk_size),
                    forward_pass_config=config,
                    keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
                )
                assert decoded.updated_state is not None
                state = decoded.updated_state
                chunk_logits = np.asarray(decoded.logits, dtype=np.float16)
                for index, (rollout, length) in enumerate(zip(batch, lengths, strict=True)):
                    output_start = max(0, length - sum(rollout.mask) - chunk_start)
                    output_end = min(chunk_size, length - chunk_start)
                    if output_start < output_end:
                        logits_by_rollout[index].append(chunk_logits[index, output_start:output_end].copy())
            result.extend(np.concatenate(chunks) for chunks in logits_by_rollout)
        return result
