"""
Hyperparameter tuning for the Sharpe LSTM with Optuna.

Each trial trains ``--seeds`` models per walk-forward fold (see
``lstm_model.make_folds``) and is scored by its validation net-of-cost Sharpe
minus equal weight's Sharpe on the same fold, averaged over folds and seeds.
Scoring against equal weight makes the edge explicit: a positive score means
the model beat simply holding every stock equally. Averaging over seeds matters: a single
Sharpe-trained network varies by more than most hyperparameter changes, so a
one-seed score mostly ranks luck. Only training-period data is used; the test
period is never seen during tuning.

Searched: tilt (how far the model may move away from equal weight, 0.1-1),
turnover penalty, rank-loss weight, concentration penalty, training window
(21 or 63 days), dropout, learning rate, weight decay. Fixed: hidden size 32,
the per-stock cap and the bond limit (lstm_model.MAX_WEIGHT / BOND_LIMIT, or
--max-weight / --bond-limit). --tilt fixes the tilt instead of searching it.

Selection: trials within ``TIE_TOLERANCE`` Sharpe of the best are treated as
ties, and among them the one with the most consistent score across folds
(lowest standard deviation) is chosen. If the best trial beats equal weight,
only tied trials that also beat it are eligible.

Trial 0 is the unregularized baseline so the result can be compared with it
directly. Unpromising trials are pruned after their first fold.

Outputs:
    data/model/best_params.json      Selected params, read by lstm_model.py
    data/results/tuning_trials.csv   Every trial with per-fold metrics

Usage:
    python tune_lstm.py
    python tune_lstm.py --trials 40 --seeds 3
    python tune_lstm.py --bond-limit 0.2 --tilt 0.5
"""

import argparse
import json
import logging

import numpy as np
import optuna

from lstm_model import (
    DEFAULT_PARAMS, PARAMS_PATH, RESULTS_DIR, SEED, decision_days, equal_weights,
    evaluate, load_inputs, make_folds, to_tensors, train,
)

TIE_TOLERANCE = 0.1     # Sharpe differences smaller than this are ties

log = logging.getLogger("tune_lstm")


def suggest_params(trial, fixed):
    """Search space; ``fixed`` holds settings given on the command line."""
    tilt = fixed["tilt"] if "tilt" in fixed else trial.suggest_float("tilt", 0.1, 1.0)
    return {
        **DEFAULT_PARAMS,
        **fixed,
        "tilt": tilt,
        "turnover_penalty": trial.suggest_float("turnover_penalty", 1e-3, 1.0, log=True),
        "rank_loss_weight": trial.suggest_float("rank_loss_weight", 1e-3, 1.0, log=True),
        "concentration_penalty": trial.suggest_float("concentration_penalty", 1e-3, 1.0, log=True),
        "train_hold_days": trial.suggest_categorical("train_hold_days", [21, 63]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }


def make_objective(data, t, folds, ew_scores, n_seeds, epochs, patience, fixed):
    def objective(trial):
        params = suggest_params(trial, fixed)
        fold_scores = []
        for k, (train_days, val_days) in enumerate(folds):
            runs = [
                train(data, params, train_days, val_days, epochs, patience,
                      seed=SEED + s, verbose=False, t=t)[1]
                for s in range(n_seeds)
            ]
            sharpe = float(np.mean([r["sharpe"] for r in runs]))
            fold_scores.append(sharpe - ew_scores[k])
            trial.set_user_attr(f"fold{k}_sharpe", sharpe)
            trial.set_user_attr(f"fold{k}_vs_equal_weight", fold_scores[-1])
            trial.set_user_attr(f"fold{k}_seed_std", float(np.std([r["sharpe"] for r in runs])))
            for key in ("turnover", "safe_weight", "epoch"):
                trial.set_user_attr(f"fold{k}_{key}", float(np.mean([r[key] for r in runs])))
            trial.report(float(np.mean(fold_scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_std", float(np.std(fold_scores)))
        log.info("trial %3d  mean val Sharpe vs equal weight %+.3f  folds %s  %s",
                 trial.number, np.mean(fold_scores), np.round(fold_scores, 3), trial.params)
        return float(np.mean(fold_scores))
    return objective


def select_trial(study):
    """Most consistent trial among those tied with the best score."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(t.value for t in done)
    tied = [t for t in done if t.value >= best - TIE_TOLERANCE]
    if best > 0:
        tied = [t for t in tied if t.value > 0]     # never trade an edge for consistency
    chosen = min(tied, key=lambda t: t.user_attrs["fold_std"])
    log.info("%d trials within %.2f of the best (%.3f); chose trial %d "
             "(score %.3f, fold std %.3f)", len(tied), TIE_TOLERANCE, best,
             chosen.number, chosen.value, chosen.user_attrs["fold_std"])
    return chosen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per fold per trial")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-weight", type=float, default=None,
                        help="Per-stock weight cap (default: lstm_model.MAX_WEIGHT)")
    parser.add_argument("--bond-limit", type=float, default=None,
                        help="Max combined SHY + TLT weight (default: lstm_model.BOND_LIMIT)")
    parser.add_argument("--tilt", type=float, default=None,
                        help="Fix the tilt away from equal weight instead of searching it")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    t = to_tensors(data)
    folds = make_folds(decision_days(data))
    for k, (tr, va) in enumerate(folds):
        log.info("fold %d  train %s -> %s | validate %s -> %s", k,
                 data["dates"][tr[0]].date(), data["dates"][tr[-1]].date(),
                 data["dates"][va[0]].date(), data["dates"][va[-1]].date())
    ew_scores = [
        evaluate(None, data, t, va, weights=equal_weights(data["tradable"][va], data["is_safe"]))["sharpe"]
        for _, va in folds
    ]
    log.info("Equal-weight validation Sharpe by fold: %s", np.round(ew_scores, 3))
    fixed = {k: getattr(args, k) for k in ("max_weight", "bond_limit", "tilt")
             if getattr(args, k) is not None}

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    # Baseline: no turnover or rank regularization, 63-day training window
    study.enqueue_trial({
        "turnover_penalty": 1e-3,
        "rank_loss_weight": 1e-3,
        "concentration_penalty": 1e-3,
        "train_hold_days": 63,
        **({} if args.tilt is not None else {"tilt": DEFAULT_PARAMS["tilt"]}),
        **{k: DEFAULT_PARAMS[k] for k in ("dropout", "lr", "weight_decay")},
    })
    study.optimize(make_objective(data, t, folds, ew_scores, args.seeds, args.epochs,
                                  args.patience, fixed),
                   n_trials=args.trials)

    study.trials_dataframe().to_csv(RESULTS_DIR / "tuning_trials.csv", index=False)
    chosen = select_trial(study)
    params = {**DEFAULT_PARAMS, **fixed, **chosen.params}
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    log.info("Baseline (trial 0) mean val Sharpe vs equal weight: %+.3f", study.trials[0].value)
    if chosen.value <= 0:
        log.warning("The chosen trial did not beat equal weight on validation (%+.3f); "
                    "a small tilt keeps the portfolio close to equal weight", chosen.value)
    log.info("Selected params -> %s\n%s", PARAMS_PATH, json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
