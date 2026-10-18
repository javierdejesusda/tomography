"""P10c: links of the range direction to knowledge and to hedging.

P10b reads a direction that separates questions whose true answer is a range
(aleatoric) from questions with one exact answer. Two links are measured on
finished runs, so no earlier config hash changes.

Knowledge: P9b reads directions that separate known from unknown questions
(epistemic). Are the two kinds of direction the same, opposite or unrelated in
one model? The cosine between the stored directions answers it layer by
layer, and so does the AUROC of each stored P9 reader on the P10 items.

Behavior: within one kind of question, does the P10b score of a test item
predict how often the model hedges (answers with a range) on the P10a run?
Across kinds the link is trivial, because the reader separates the kinds and
the behavior differs by kind.

The inputs `readers.npz` and `items.json` are local only (export drops them),
so this experiment runs where the full run directories exist.
"""

from dataclasses import dataclass
from typing import Final

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.p10b_reading import TEMPLATES
from tomography.p9.reading import READING_METHODS

READER_PARTS: Final = 2


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P10c settings: the finished runs to link and the reader to use.

    Attributes:
        p10b_run: Path of a finished P10b run of the same model.
        knowledge_run: Path of a finished P9b run of the same model, or None
            to skip the knowledge link.
        behavior_run: Path of a finished P10a run of the same model, or None
            to skip the behavior link.
        reader: The P10b reader, named "template/method".
        n_boot: Resamples of every bootstrap interval.
        n_random: Random unit directions of every random band.
        batch_size: Prompts per forward pass.
        offline: Whether the questions are read from the cache only.
    """

    experiment: str = "p10c"
    model: str = "gpt2"
    p10b_run: str
    knowledge_run: str | None = None
    behavior_run: str | None = None
    reader: str = "plain/logistic"
    n_boot: int = 1000
    n_random: int = 200
    batch_size: int = 8
    offline: bool = True

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If neither `knowledge_run` nor `behavior_run` is
                set, `reader` is not a "template/method" name of P10b, or a
                count is below 1.
        """
        if self.knowledge_run is None and self.behavior_run is None:
            raise ValueError(
                "set at least one of knowledge_run and behavior_run"
            )
        parts = self.reader.split("/")
        if (
            len(parts) != READER_PARTS
            or parts[0] not in TEMPLATES
            or parts[1] not in READING_METHODS
        ):
            raise ValueError(
                f"reader must be 'template/method' with a template in "
                f"{TEMPLATES} and a method in {READING_METHODS}, got "
                f"{self.reader!r}"
            )
        for name in ("n_boot", "n_random", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
