# ruff: noqa: I001
import json
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import cattrs
from loguru import logger
from typer import Argument, Option, Typer

from .model import Model
from .vendors.lalamo import LalamoModel
from .dataset import load_conversations
from .scoring import score
from .rollouts import Rollout, hash_folder_with_edittimes, load_rollouts, save_rollouts

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def run(
    reference: Annotated[Path, Argument(help="Lalamo reference model path")],
    students: Annotated[list[str], Argument(help="Student model paths")],
    dataset: Annotated[str, Option(help="Hugging Face conversation dataset")] = "trymirai/kl-mixture",
    cache_path: Annotated[Path, Option(help="Teacher trace cache directory")] = Path("teacher-cache"),
) -> None:
    conversations = load_conversations(dataset)
    logger.info("Loaded {} conversations", len(conversations))

    reference_model = LalamoModel.load(reference.expanduser())
    models = {name: Model.init(name, reference_model.codec.config) for name in students}
    generation_config = reference_model.model.config.generation_config
    logger.info("Loaded the reference and {} students", len(models))

    prompts = reference_model.tokenize(conversations)
    if any(model.tokenize(conversations) != prompts for model in models.values()):
        raise ValueError("model tokenization differs from the reference")
    prompts = [prompt for prompt in prompts if len(prompt) <= 32_768]
    logger.info("Kept {} prompts of at most 32k tokens", len(prompts))

    cache_key = (hash_folder_with_edittimes(reference_model.path), prompts, cattrs.unstructure(generation_config))
    rollout_path = cache_path / f"{sha256(json.dumps(cache_key).encode()).hexdigest()}.json"
    reference_rollouts: list[Rollout] | None = load_rollouts(rollout_path)
    if reference_rollouts is None:
        logger.info("Generating {} rollouts", len(prompts))
        reference_rollouts = reference_model.generate(prompts, generation_config, seed=0, token_limit=256)
        save_rollouts(rollout_path, reference_rollouts)

    logger.info("Computing teacher logits")
    teacher_logits = reference_model.logits(reference_rollouts)
    scores = {}
    for student_name in students:
        logger.info("Scoring {}", student_name)
        scores[student_name] = score(models[student_name], teacher_logits, reference_rollouts, generation_config)
        logger.info(f"Model {student_name} got {scores[student_name]}")
        logger.info(
            "Model {} generates with {:.2f} GB of weights", student_name, models[student_name].generator_bytes() / 1e9
        )

    logger.info("done.")


if __name__ == "__main__":
    app()
