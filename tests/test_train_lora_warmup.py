from scripts.train_lora import FUSION_LORA_WARMUP, compute_warmup_steps, default_warmup_steps


def test_no_frac_passes_through_raw_steps():
    assert compute_warmup_steps(
        warmup_steps=250, warmup_frac=None,
        dataset_len=5400, batch_size=4, accumulate_grad_batches=1, epochs=20, steps=10_000,
    ) == 250


def test_frac_of_epoch_mode_is_one_epoch_at_5pct_20_epochs():
    # goa DoRA recipe: 5400 crops, bs4 -> 1350 steps/epoch x 20 epochs = 27000 total.
    # 5% of a 20-epoch run is exactly 1 epoch, independent of corpus size.
    steps = compute_warmup_steps(
        warmup_steps=0, warmup_frac=0.05,
        dataset_len=5400, batch_size=4, accumulate_grad_batches=1, epochs=20, steps=10_000,
    )
    assert steps == 1350


def test_frac_of_steps_mode_uses_steps_directly_when_no_epochs():
    steps = compute_warmup_steps(
        warmup_steps=0, warmup_frac=0.05,
        dataset_len=5400, batch_size=4, accumulate_grad_batches=1, epochs=None, steps=20_000,
    )
    assert steps == 1000


def test_frac_accounts_for_grad_accumulation():
    # accumulate_grad_batches=2 halves the optimizer-step rate: 1350//2=675 steps/epoch
    # x20 = 13500 total, 5% = 675.
    steps = compute_warmup_steps(
        warmup_steps=0, warmup_frac=0.05,
        dataset_len=5400, batch_size=4, accumulate_grad_batches=2, epochs=20, steps=10_000,
    )
    assert steps == 675


def test_frac_overrides_raw_warmup_steps_when_both_given():
    steps = compute_warmup_steps(
        warmup_steps=999, warmup_frac=0.05,
        dataset_len=5400, batch_size=4, accumulate_grad_batches=1, epochs=20, steps=10_000,
    )
    assert steps == 1350


def test_fusion_lora_defaults_to_the_momentum_window():
    assert default_warmup_steps(None, "fusion", False) == FUSION_LORA_WARMUP == 20


def test_default_stays_zero_off_fusion_lora():
    assert default_warmup_steps(None, "adamw", False) == 0
    assert default_warmup_steps(None, "lion", False) == 0
    assert default_warmup_steps(None, "fusion", True) == 0


def test_explicit_zero_restores_the_old_behaviour():
    assert default_warmup_steps(0, "fusion", False) == 0
    assert default_warmup_steps(250, "adamw", False) == 250
