"""
Hyperparameter tuning for the Sharpe LSTM with Optuna.

Searches the turnover and concentration penalty strengths, the single-name
weight cap, and core LSTM settings. Each trial trains one model per
walk-forward fold (see ``lstm_model.make_folds``) and is scored by the mean
validation net-of-cost Sharpe across folds. Only training-period data is used;
the test period is never seen during tuning.

Trial 0 is the unregularized baseline (no penalties, 15% cap) so the tuned
result can be compared with it directly. Unpromising trials are pruned after
their first fold.

Outputs:
    data/model/best_params.json      Best params, read by lstm_model.py
    data/results/tuning_trials.csv   Every trial with per-fold metrics

Usage:
    python tune_lstm.py
    python tune_lstm.py --trials 60 --epochs 40
"""

import argparse
import json
import logging

import numpy as np
import optuna

from lstm_model import (
    DEFAULT_PARAMS, PARAMS_PATH, RESULTS_DIR, SEED, decision_days, load_inputs,
    make_folds, train,
)

log = logging.getLogger("tune_lstm")


def suggest_params(trial):
    return {
        "turnover_penalty": trial.suggest_float("turnover_penalty", 1e-3, 2.0, log=True),
        "concentration_penalty": trial.suggest_float("concentration_penalty", 1e-3, 5.0, log=True),
        "max_weight": trial.suggest_categorical("max_weight", [0.10, 0.15, 0.25, 1.0]),
        "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": DEFAULT_PARAMS["batch_size"],
    }


def make_objective(data, folds, epochs, patience):
    def objective(trial):
        params = suggest_params(trial)
        scores = []
        for k, (train_days, val_days) in enumerate(folds):
            _, best = train(data, params, train_days, val_days, epochs, patience, verbose=False)
            scores.append(best["sharpe"])
            for key in ("sharpe", "turnover", "hhi", "max_weight", "epoch"):
                trial.set_user_attr(f"fold{k}_{key}", best[key])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        log.info("trial %3d  mean val net Sharpe %.3f  folds %s  %s",
                 trial.number, np.mean(scores), np.round(scores, 3), trial.params)
        return float(np.mean(scores))
    return objective


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    folds = make_folds(decision_days(data))
    for k, (tr, va) in enumerate(folds):
        log.info("fold %d  train %s -> %s | validate %s -> %s", k,
                 data["dates"][tr[0]].date(), data["dates"][tr[-1]].date(),
                 data["dates"][va[0]].date(), data["dates"][va[-1]].date())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    # Baseline: the unregularized model from before tuning
    study.enqueue_trial({
        **{k: DEFAULT_PARAMS[k] for k in ("max_weight", "hidden", "dropout", "lr", "weight_decay")},
        "turnover_penalty": 1e-3,
        "concentration_penalty": 1e-3,
    })
    study.optimize(make_objective(data, folds, args.epochs, args.patience), n_trials=args.trials)

    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "tuning_trials.csv", index=False)

    best = {**DEFAULT_PARAMS, **study.best_params}
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    baseline = study.trials[0]
    log.info("Baseline (trial 0) mean val net Sharpe: %.3f", baseline.value)
    log.info("Best (trial %d) mean val net Sharpe: %.3f", study.best_trial.number, study.best_value)
    log.info("Best params -> %s\n%s", PARAMS_PATH, json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
