from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.data_loader import load_modeling_data
from src.preprocessing.preprocessor import (
    BASELINE_FEATURES,
    build_preprocessor,
)


RANDOM_STATE = 42
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_JSON = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "model_cross_validation_2023.json"
)

REPORT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "model_cross_validation_2023.csv"
)


def build_candidates():
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=100,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=100,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_fold_metrics(
    y_true,
    y_pred,
    y_proba,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    y_risk = (y_true == 0).astype(int)
    proba_risk = 1.0 - y_proba

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_proba,
            )
        ),
        "pr_auc_nao_alfabetizado": float(
            average_precision_score(
                y_risk,
                proba_risk,
            )
        ),
        "recall_nao_alfabetizado": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
            )
        ),
        "f1_nao_alfabetizado": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
            )
        ),
    }


def main():
    print(
        "Carregando exclusivamente dados de 2023..."
    )

    df = load_modeling_data(2023)

    X = df[BASELINE_FEATURES]
    y = df["alfabetizado"].astype(int)
    groups = df["id_municipio"].astype(str)

    print(f"Registros: {len(df):,}")
    print(
        f"Municípios: {groups.nunique():,}"
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    all_results = []

    for model_name, classifier in (
        build_candidates().items()
    ):
        print(
            f"\n{'=' * 60}"
        )
        print(
            f"Modelo: {model_name}"
        )
        print(
            f"{'=' * 60}"
        )

        fold_results = []

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            cv.split(
                X,
                y,
                groups=groups,
            ),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            groups_train = set(
                groups.iloc[
                    train_idx
                ].unique()
            )

            groups_val = set(
                groups.iloc[
                    val_idx
                ].unique()
            )

            overlap = (
                groups_train
                & groups_val
            )

            assert not overlap

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            include_uf=False
                        ),
                    ),
                    (
                        "classifier",
                        classifier,
                    ),
                ]
            )

            start = perf_counter()

            pipeline.fit(
                X_train,
                y_train,
            )

            seconds = (
                perf_counter()
                - start
            )

            y_pred = pipeline.predict(
                X_val
            )

            y_proba = (
                pipeline.predict_proba(
                    X_val
                )[:, 1]
            )

            metrics = (
                calculate_fold_metrics(
                    y_true=y_val,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )
            )

            result = {
                "modelo": model_name,
                "fold": fold,
                "registros_treino": int(
                    len(train_idx)
                ),
                "registros_validacao": int(
                    len(val_idx)
                ),
                "municipios_treino": int(
                    len(groups_train)
                ),
                "municipios_validacao": int(
                    len(groups_val)
                ),
                "municipios_overlap": int(
                    len(overlap)
                ),
                "treinamento_segundos": float(
                    seconds
                ),
                **metrics,
            }

            fold_results.append(
                result
            )

            all_results.append(
                result
            )

            print(
                f"Fold {fold}: "
                f"BA={metrics['balanced_accuracy']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"PR-AUC risco={metrics['pr_auc_nao_alfabetizado']:.4f} | "
                f"Recall risco={metrics['recall_nao_alfabetizado']:.4f}"
            )

    results_df = pd.DataFrame(
        all_results
    )

    summary = (
        results_df
        .groupby("modelo")
        .agg(
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            pr_auc_risk_mean=(
                "pr_auc_nao_alfabetizado",
                "mean",
            ),
            pr_auc_risk_std=(
                "pr_auc_nao_alfabetizado",
                "std",
            ),
            recall_risk_mean=(
                "recall_nao_alfabetizado",
                "mean",
            ),
            recall_risk_std=(
                "recall_nao_alfabetizado",
                "std",
            ),
            f1_risk_mean=(
                "f1_nao_alfabetizado",
                "mean",
            ),
            f1_risk_std=(
                "f1_nao_alfabetizado",
                "std",
            ),
            training_seconds_mean=(
                "treinamento_segundos",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "roc_auc_mean",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 60
        + "\nRESUMO CROSS-VALIDATION\n"
        + "=" * 60
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "ano": 2023,
        "n_splits": N_SPLITS,
        "group_variable": (
            "id_municipio"
        ),
        "features": BASELINE_FEATURES,
        "folds": all_results,
        "summary": (
            summary.to_dict(
                orient="records"
            )
        ),
    }

    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    results_df.to_csv(
        REPORT_CSV,
        index=False,
    )

    print(
        "\nResultados salvos em:"
    )

    print(
        REPORT_JSON.relative_to(
            PROJECT_ROOT
        )
    )

    print(
        REPORT_CSV.relative_to(
            PROJECT_ROOT
        )
    )


if __name__ == "__main__":
    main()