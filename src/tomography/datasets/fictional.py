"""Seeded fictional names matched to real names on token count.

A real name and an invented one differ in how many tokens they cost, which
a probe could pick up instead of whether the model knows the entity. The
generator invents names with the same token count and the same number of
words as the real subjects, as a control for that name artifact.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np

from tomography.datasets import popqa
from tomography.datasets.records import QAExample

_CONSONANTS = "bcdfghjklmnprstvz"
_VOWELS = "aeiou"
_MIN_SYLLABLE_CAP = 4
_MIN_TOKENS = 2


def generate_fictional(
    real_items: Sequence[QAExample],
    *,
    count_tokens: Callable[[str], int],
    seed: int,
    templates: Mapping[str, str] = popqa.TEMPLATES,
    max_attempts: int = 5000,
) -> list[QAExample]:
    """Builds a fictional counterpart for each real question item.

    The counterpart asks the same relation with the shared template about an
    invented subject whose token and word count match the real one. It
    shares the real item's split, so a held-out set stays matched.

    Args:
        real_items: Items whose meta holds "relation" and "subject", such as
            the output of `load_popqa`.
        count_tokens: Number of tokens of a name as the model sees it in the
            question.
        seed: Seeds the name draws.
        templates: Question template per relation, with one "{}" slot.
        max_attempts: Candidates tried per subject before giving up.

    Returns:
        One "fictional" item per real item, in order, with the real item's
        uid in "matched_uid" and the shared token count in "name_tokens".

    Raises:
        ValueError: If a relation has no template, a subject has fewer than
            2 tokens (drop single-token subjects first) or a subject cannot
            be matched within `max_attempts`.
    """
    for item in real_items:
        if item.meta["relation"] not in templates:
            raise ValueError(
                f"{item.uid}: no template for relation "
                f"{item.meta['relation']!r}"
            )
    names = generate_names(
        [str(item.meta["subject"]) for item in real_items],
        count_tokens=count_tokens,
        seed=seed,
        max_attempts=max_attempts,
    )
    return [
        QAExample(
            uid=f"fictional:{item.uid}",
            question=templates[str(item.meta["relation"])].replace("{}", name),
            answers=(),
            kind="fictional",
            split=item.split,
            meta={
                "relation": item.meta["relation"],
                "subject": name,
                "popularity": None,
                "source": "generated",
                "matched_uid": item.uid,
                "name_tokens": count_tokens(name),
            },
        )
        for item, name in zip(real_items, names, strict=True)
    ]


def generate_names(
    targets: Sequence[str],
    *,
    count_tokens: Callable[[str], int],
    seed: int,
    max_attempts: int = 5000,
) -> list[str]:
    """Invents one name per target with the same token and word count.

    Names are built from consonant-vowel syllables, with each word
    capitalized. The total syllable count is drawn up to three times the
    target's token count, so long names can be matched. Each target draws
    from its own stream, so the result depends only on the seed, the target
    list and `count_tokens`. A name is never a target and never repeats an
    earlier name. Callers must drop targets of a single token first: a
    one-token name would be a stub such as "He".

    Args:
        targets: Real names to match, such as "George Rankin".
        count_tokens: Number of tokens of a name as the model sees it in the
            question, for example a tokenizer applied to the name with its
            leading space.
        seed: Seeds the draws.
        max_attempts: Candidates tried per target before giving up.

    Returns:
        One invented name per target, in order.

    Raises:
        ValueError: If a target has fewer than 2 tokens or no candidate
            matches a target within `max_attempts`.
    """
    taken = {target.lower() for target in targets}
    names = []
    for index, target in enumerate(targets):
        goal = count_tokens(target)
        if goal < _MIN_TOKENS:
            raise ValueError(
                f"{target!r} has {goal} token(s); at least {_MIN_TOKENS} "
                "are needed"
            )
        words = max(1, len(target.split()))
        cap = max(_MIN_SYLLABLE_CAP * words, 3 * goal)
        rng = np.random.default_rng([seed, index])
        for _ in range(max_attempts):
            name = _draw_name(rng, words, cap)
            if name.lower() not in taken and count_tokens(name) == goal:
                taken.add(name.lower())
                names.append(name)
                break
        else:
            raise ValueError(
                f"no name with {goal} tokens in {words} words like "
                f"{target!r} in {max_attempts} attempts"
            )
    return names


def _draw_name(rng: np.random.Generator, words: int, cap: int) -> str:
    total = int(rng.integers(words, cap + 1))
    extra = rng.multinomial(total - words, [1 / words] * words)
    return " ".join(_draw_word(rng, 1 + int(more)) for more in extra)


def _draw_word(rng: np.random.Generator, syllables: int) -> str:
    letters = []
    for _ in range(syllables):
        letters.append(_CONSONANTS[int(rng.integers(len(_CONSONANTS)))])
        letters.append(_VOWELS[int(rng.integers(len(_VOWELS)))])
    return "".join(letters).capitalize()
