# ruff: noqa: I001
from .model import Model
from .rollouts import Rollout
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from lalamo.models.language_model import GenerationConfig


def log_probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    maximum = logits.max(axis=-1, keepdims=True)
    return logits - maximum - np.log(np.exp(logits - maximum).sum(axis=-1, keepdims=True))


def expected_sampling_disagreement(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    rollout: Rollout,
    generation_config: GenerationConfig,
) -> np.ndarray:
    policy = generation_config.default_policy()
    if not policy.has_count_penalties:
        output_tokens = len(student_logits)
        padding = ((0, -output_tokens % 256), (0, 0))
        student_rows = jnp.pad(jnp.asarray(student_logits, dtype=jnp.float32), padding)
        teacher_rows = jnp.pad(jnp.asarray(teacher_logits, dtype=jnp.float32), padding)
        student_probs = jax.nn.softmax(jax.vmap(policy.process_logits)(student_rows))
        teacher_probs = jax.nn.softmax(jax.vmap(policy.process_logits)(teacher_rows))
        return np.asarray(jnp.abs(student_probs - teacher_probs).sum(axis=-1)[:output_tokens] / 2)

    output_start = len(rollout.tokens) - sum(rollout.mask)
    prompt = jnp.asarray(rollout.tokens[:output_start], dtype=jnp.int32)
    policy = policy.with_prompt_token_counts(prompt, jnp.asarray(output_start), student_logits.shape[-1])

    disagreements = []
    for student_row, teacher_row, token in zip(
        student_logits, teacher_logits, rollout.tokens[output_start:], strict=True
    ):
        student_probs = jax.nn.softmax(policy.process_logits(jnp.asarray(student_row, dtype=jnp.float32)))
        teacher_probs = jax.nn.softmax(policy.process_logits(jnp.asarray(teacher_row, dtype=jnp.float32)))
        disagreements.append(jnp.abs(student_probs - teacher_probs).sum() / 2)
        policy = policy.with_next_token_count(jnp.asarray(token))
    return np.asarray(jnp.stack(disagreements))


@dataclass(frozen=True)
class Score:
    kl: float
    tv: float
    esd: float

    def __str__(self) -> str:
        return f"KL {self.kl:>8.3g}  TV {self.tv:>8.3g}  ESD {self.esd:>8.3g}"


def score(
    student: Model, all_teacher_logits: list[np.ndarray], rollouts: list[Rollout], generation_config: GenerationConfig
) -> Score:
    all_student_logits = student.logits(rollouts)

    total_kl, total_tv, total_esd, total_tokens = 0.0, 0.0, 0.0, 0
    for rollout, teacher_logits, student_logits in zip(rollouts, all_teacher_logits, all_student_logits, strict=True):
        teacher_logp = log_probabilities_from_logits(teacher_logits)
        student_logp = log_probabilities_from_logits(student_logits)
        teacher_prob = np.exp(teacher_logp)
        student_prob = np.exp(student_logp)
        total_kl += float((teacher_prob * (teacher_logp - student_logp)).sum())
        total_tv += float(np.abs(teacher_prob - student_prob).sum() / 2)
        total_esd += float(
            expected_sampling_disagreement(student_logits, teacher_logits, rollout, generation_config).sum(),
        )
        total_tokens += sum(rollout.mask)

    return Score(kl=total_kl / total_tokens, tv=total_tv / total_tokens, esd=total_esd / total_tokens)
