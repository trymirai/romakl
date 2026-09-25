# ruff: noqa: I001
import json
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import cattrs
from loguru import logger
from typer import Argument, Option, Typer

from .model import Model
from .dataset import load_conversations
from .scoring import score
from .rollouts import Rollout, hash_folder_with_edittimes, load_rollouts, save_rollouts

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def run(
    reference: Annotated[str, Argument(help="Lalamo reference model path")],
    students: Annotated[list[str], Argument(help="Student model paths")],
    dataset: Annotated[str, Option(help="Hugging Face conversation dataset")] = "trymirai/kl-mixture",
    cache_path: Annotated[Path, Option(help="Teacher trace cache directory")] = Path("teacher-cache"),
) -> None:
    conversations = load_conversations(dataset)
    logger.info("Loaded {} conversations", len(conversations))

    models = {name: Model.init(name) for name in [reference, *students]}

    reference_model = models[reference]
    generation_params = reference_model.generation_params

    prompts = reference_model.tokenize(conversations)
    if any(model.tokenize(conversations) != prompts for model in models.values()):
        raise ValueError("model tokenization differs from the reference")

    cache_data = (hash_folder_with_edittimes(reference_model.path), prompts, cattrs.unstructure(generation_params))
    rollout_path = cache_path / f"{sha256(json.dumps(cache_data).encode()).hexdigest()}.json"
    reference_rollouts: list[Rollout] | None = load_rollouts(rollout_path)
    if reference_rollouts is None:
        reference_rollouts = reference_model.generate(prompts, generation_params)
        save_rollouts(rollout_path, reference_rollouts)

    scores = {}
    for student_name in students:
        scores[student_name] = score(models[student_name], reference_model, reference_rollouts, generation_params)
        logger.info(f"Model {student_name} got {scores[student_name]}")

    logger.info("done.")


if __name__ == "__main__":
    app()
