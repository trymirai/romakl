import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cattrs


@dataclass(frozen=True)
class Rollout:
    tokens: list[int]
    mask: list[int]

    def __post_init__(self) -> None:
        assert len(self.mask) == len(self.tokens), "mask must have one entry per token"
        assert set(self.mask) <= {0, 1}, "mask must be binary"
        assert self.mask == sorted(self.mask), "mask must be prompt zeros followed by continuation ones"
        assert 0 < sum(self.mask) < len(self.mask), "rollout must have both a prompt and a continuation"


def hash_folder_with_edittimes(path: Path | str) -> str:
    path = Path(path).resolve()
    path.stat()
    if path.is_file():  # noqa: SIM108
        files = [path]
    else:
        files = sorted(path.rglob("*"))
    edits = [
        (str(file.relative_to(path)), file.stat().st_mtime_ns)
        for file in files
        if file.is_file() and not any(part.startswith(".") for part in file.relative_to(path).parts)
    ]
    return sha256(json.dumps((str(path), edits)).encode()).hexdigest()


def load_rollouts(cache_path: Path | str) -> list[Rollout] | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    return cattrs.structure(json.loads(path.read_text()), list[Rollout])


def save_rollouts(cache_path: Path | str, rollouts: list[Rollout]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cattrs.unstructure(rollouts)))
