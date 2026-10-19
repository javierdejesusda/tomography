from tomography.site import store


def test_post_trained_means_qwen3_without_the_base_suffix() -> None:
    assert store.is_post_trained("qwen3-1.7b")
    assert not store.is_post_trained("qwen3-1.7b-base")
    assert not store.is_post_trained("gpt2-xl")


def test_models_sort_in_paper_order_with_unknown_keys_last() -> None:
    keys = ["qwen3-4b", "zzz", "gpt2-xl", "gpt2", "qwen3-0.6b-base"]
    assert sorted(keys, key=store.model_rank) == [
        "gpt2",
        "gpt2-xl",
        "qwen3-0.6b-base",
        "qwen3-4b",
        "zzz",
    ]
