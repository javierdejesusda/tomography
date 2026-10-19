"""Models, judges and datasets the page uses, with their pins and licenses.

Every row repeats a pin that the code reads: `model.py` for the checkpoints,
`judges.py` for the judges and the `SOURCE` of each module under `datasets/`.
A test compares each row with its pin, so a revision here cannot drift from
the one the experiments load. Licenses are those the sources state.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Asset:
    """One model, judge or dataset.

    Attributes:
        key: Short identifier.
        name: Name as the page writes it.
        used: What the page uses it for.
        cite: Bibliography keys of the works to credit.
        location: Hub ID or source address.
        revision: First 12 characters of the pinned commit, or the digest of
            a pinned file, or a description of the snapshot rule.
        license: License as the source states it.
    """

    key: str
    name: str
    used: str
    cite: tuple[str, ...]
    location: str
    revision: str
    license: str


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """The pin of one checkpoint of `model.py`.

    Attributes:
        repo_id: Hub repository id.
        revision: Full commit SHA of the weights.
        license: License as the Hub card states it.
    """

    repo_id: str
    revision: str
    license: str


CHECKPOINTS = {
    "gpt2": Checkpoint(
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "Modified MIT",
    ),
    "gpt2-medium": Checkpoint(
        "openai-community/gpt2-medium",
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "Modified MIT",
    ),
    "gpt2-large": Checkpoint(
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
        "Modified MIT",
    ),
    "gpt2-xl": Checkpoint(
        "openai-community/gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
        "Modified MIT",
    ),
    "qwen3-0.6b-base": Checkpoint(
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "Apache-2.0",
    ),
    "qwen3-0.6b": Checkpoint(
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "Apache-2.0",
    ),
    "qwen3-1.7b-base": Checkpoint(
        "Qwen/Qwen3-1.7B-Base",
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "Apache-2.0",
    ),
    "qwen3-1.7b": Checkpoint(
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Apache-2.0",
    ),
    "qwen3-4b-base": Checkpoint(
        "Qwen/Qwen3-4B-Base",
        "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "Apache-2.0",
    ),
    "qwen3-4b": Checkpoint(
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "Apache-2.0",
    ),
}

PER_CHECKPOINT = "one per checkpoint, [[table:checkpoints]]"

ASSETS = (
    Asset(
        "gpt2",
        "GPT-2",
        "four checkpoints, small to XL",
        ("radford2019language",),
        "openai-community/gpt2, gpt2-medium, gpt2-large, gpt2-xl",
        PER_CHECKPOINT,
        "Modified MIT",
    ),
    Asset(
        "qwen3",
        "Qwen3",
        "six checkpoints, base and post-trained at three sizes",
        ("yang2025qwen3",),
        "Qwen/Qwen3-0.6B-Base, -0.6B, -1.7B-Base, -1.7B, -4B-Base, -4B",
        PER_CHECKPOINT,
        "Apache-2.0",
    ),
    Asset(
        "siebert",
        "SiEBERT",
        "judge of the sentiment of generated text",
        ("hartmann2023more",),
        "siebert/sentiment-roberta-large-english",
        "74cea614e245",
        "none declared",
    ),
    Asset(
        "distilbert",
        "DistilBERT SST-2",
        "second judge of the sentiment of generated text",
        ("sanh2019distilbert",),
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "714eb0fa89d2",
        "apache-2.0",
    ),
    Asset(
        "sst2",
        "SST-2",
        "sentiment pairs",
        ("socher2013recursive",),
        "stanfordnlp/sst2",
        "8d51e7e4887a",
        "Unknown (cite Socher et al., 2013)",
    ),
    Asset(
        "ethics",
        "ETHICS",
        "utility and morality pairs",
        ("hendrycks2021aligning",),
        "people.eecs.berkeley.edu/~hendrycks/ethics.tar",
        "SHA-256 40acbf1ac0da",
        "MIT",
    ),
    Asset(
        "repe_facts",
        "True and false facts of the representation engineering repository",
        "truth and honesty statements",
        (
            "zou2025representationengineeringtopdownapproach",
            "azaria2023internal",
        ),
        "github.com/andyzoujm/representation-engineering",
        "5455d8a375d5",
        "MIT (granted by the representation-engineering repository)",
    ),
    Asset(
        "geometry_of_truth",
        "Geometry of Truth",
        "out-of-distribution truth sets",
        ("marks2024geometry",),
        "github.com/saprmarks/geometry-of-truth",
        "5d1c630c44f7",
        "None stated (runtime download only)",
    ),
    Asset(
        "commonclaim",
        "CommonClaim",
        "common claims set, as shipped with Geometry of Truth",
        ("casper2023explore",),
        "github.com/saprmarks/geometry-of-truth",
        "5d1c630c44f7",
        "None stated (runtime download only)",
    ),
    Asset(
        "truthfulqa",
        "TruthfulQA",
        "multiple-choice truthfulness questions",
        ("lin2022truthfulqa",),
        "truthfulqa/truthful_qa",
        "741b8276f2d1",
        "Apache-2.0",
    ),
    Asset(
        "arc",
        "ARC",
        "multiple-choice questions for a second reader source",
        ("clark2018think",),
        "allenai/ai2_arc",
        "210d026faf99",
        "CC-BY-SA-4.0",
    ),
    Asset(
        "popqa",
        "PopQA",
        "known and unknown questions, subject popularity",
        ("mallen2023when",),
        "akariasai/PopQA",
        "098765c79ea1",
        "no license on the dataset card (code repository: MIT)",
    ),
    Asset(
        "triviaqa",
        "TriviaQA",
        "known and unknown questions",
        ("joshi2017triviaqa",),
        "mandarjoshi/trivia_qa",
        "0f7faf33a390",
        "unknown on the dataset card, Apache-2.0 on the GitHub repository",
    ),
    Asset(
        "kuq",
        "KUQ",
        "known and unknown questions",
        ("amayuelas2024knowledge",),
        "amayuelas/KUQ",
        "f99b53aa226d",
        "MIT",
    ),
    Asset(
        "falseqa",
        "FalseQA",
        "questions with false premises",
        ("hu2023wont",),
        "github.com/thunlp/FalseQA",
        "9a9729974d8c",
        "no license: runtime download only",
    ),
    Asset(
        "synthal",
        "SyntHal",
        "synthetic unanswerable questions",
        ("lu2026relational",),
        "github.com/yuetianluNLP/SyntHal",
        "c4fe4cecf41b",
        "research and review terms, no permissive license",
    ),
    Asset(
        "nq_open",
        "NQ-open",
        "known and unknown questions",
        ("kwiatkowski2019natural", "lee2019latent"),
        "google-research-datasets/nq_open",
        "5dd9790a8300",
        "CC-BY-SA-3.0",
    ),
    Asset(
        "wikidata",
        "Wikidata",
        "quantities and award questions",
        ("vrandecic2014wikidata",),
        "query.wikidata.org/sparql",
        "snapshot date and SHA-256 per item",
        "CC0",
    ),
    Asset(
        "wikitext103",
        "WikiText-103",
        "held-out passages for perplexity",
        ("merity2017pointer",),
        "Salesforce/wikitext",
        "b08601e04326",
        "CC-BY-SA-3.0 and GFDL",
    ),
)
