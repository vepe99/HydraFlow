"""Stage 4: hyperparameter tuning with Optuna.

Runs a (by default multi-objective) study minimizing RMSE and calibration error. The dataset is
loaded and preprocessed once; each trial applies its sampled hyperparameters onto a copy of the
config, builds a fresh workflow, trains for a short budget (``tuning.n_epochs``), and scores the
validation split. Generalizes ``main_hyperparameter_tuning_nocompositional_rotationcurve_agama.py``
but composes the config via ``compose`` rather than nesting ``@hydra.main``.

The search space (``conf/tuning/search_space``) maps dotted config paths to sampling specs:
``{type: int|float|categorical, low, high, step, log, choices}``.
"""

from __future__ import annotations

import copy
import os

import numpy as np

from hydraflow.pipeline import io
from hydraflow.pipeline._app import make_cli
from hydraflow.pipeline.workflow import build_workflow
from hydraflow.preprocessing.registry import build_pipeline
from hydraflow.utils.logging import get_logger
from hydraflow.utils.paths import get_run_dir
from hydraflow.utils.seed import seed_everything

log = get_logger(__name__)


def _suggest(trial, name, spec):
    t = spec["type"]
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if t == "float":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]),
            step=spec.get("step"), log=bool(spec.get("log", False)),
        )
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown search-space type '{t}' for '{name}'")


def _objective(trial, base_cfg, train_data, val_data, param_names):
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(base_cfg)
    for path, spec in base_cfg.tuning.search_space.items():
        OmegaConf.update(cfg, path, _suggest(trial, path, spec), force_add=True)

    workflow = build_workflow(cfg)
    workflow.fit_offline(
        train_data,
        validation_data=val_data,
        epochs=int(cfg.tuning.n_epochs),
        batch_size=int(cfg.training.batch_size),
        verbose=0,
    )

    from bayesflow.diagnostics import metrics as bf_metrics

    posterior = workflow.sample(
        num_samples=int(cfg.inference.num_samples),
        conditions=val_data,
        batch_size=int(cfg.inference.batch_size),
    )
    rmse = bf_metrics.root_mean_squared_error(
        estimates=posterior, targets=val_data, variable_keys=param_names
    )
    cal = bf_metrics.calibration_error(
        estimates=posterior, targets=val_data, variable_keys=param_names
    )
    rmse_mean = float(np.mean(rmse["values"]))
    cal_mean = float(np.mean(cal["values"]))

    if len(base_cfg.tuning.directions) == 1:
        return rmse_mean
    return rmse_mean, cal_mean


def run_tuning(cfg):
    import optuna

    rng = seed_everything(cfg.seed)
    run_dir = get_run_dir()

    # Load + preprocess once; reuse across trials.
    data = io.load_dataset(os.path.join(cfg.data.data_dir, cfg.data.dataset_name))
    pipeline = build_pipeline(cfg.preprocessing)
    train_data, val_data = pipeline.fit_transform(data, rng)
    param_names = list(cfg.adapter.inference_variables)

    os.makedirs(cfg.tuning.storage_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(cfg.tuning.storage_dir, cfg.tuning.study_name + '.db')}"
    study = optuna.create_study(
        study_name=cfg.tuning.study_name,
        storage=storage,
        directions=list(cfg.tuning.directions),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: _objective(trial, cfg, train_data, val_data, param_names),
        n_trials=int(cfg.tuning.n_trials),
    )

    _report(study, cfg, run_dir)
    return study


def _report(study, cfg, run_dir) -> None:
    import json

    if len(cfg.tuning.directions) == 1:
        best = [{"number": study.best_trial.number, "values": study.best_trial.value,
                 "params": study.best_trial.params}]
    else:
        best = [
            {"number": t.number, "values": t.values, "params": t.params}
            for t in study.best_trials
        ]
    with open(os.path.join(run_dir, "best_trials.json"), "w") as f:
        json.dump(best, f, indent=2)
    log.info("Tuning complete: %d Pareto/best trial(s). See %s", len(best), run_dir)


cli = make_cli(run_tuning)


if __name__ == "__main__":
    cli()
