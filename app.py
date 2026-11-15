import math
import time

import numpy as np
import pandas as pd
import streamlit as st
from mealpy import GA, HBA, PSO, FloatVar
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def make_knn_pipeline(n_neighbors):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def evaluate_knn(x_train, x_test, y_train, y_test, n_neighbors, n_classes):
    model = make_knn_pipeline(n_neighbors)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="binary" if n_classes == 2 else "weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="binary" if n_classes == 2 else "weighted", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, average="binary" if n_classes == 2 else "weighted", zero_division=0
        ),
    }

    probabilities = model.predict_proba(x_test)
    if n_classes == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_test, probabilities[:, 1])
    else:
        metrics["ROC-AUC"] = roc_auc_score(
            y_test,
            probabilities,
            labels=model.named_steps["knn"].classes_,
            multi_class="ovr",
            average="weighted",
        )

    return metrics, confusion_matrix(y_test, predictions)


st.set_page_config(
    page_title="HBA-KNN Metaheuristic Feature Optimization Suite",
    page_icon="🦡",
    layout="wide",
)

st.title("HBA-KNN Metaheuristic Feature Optimization Suite")
st.markdown(
    "A wrapper feature-selection and benchmarking suite using Honey Badger Optimization "
    "(HBA) with a K-Nearest Neighbors (KNN) classifier."
)

st.sidebar.header("Experimental controls")
k_neighbors = st.sidebar.slider("KNN k", min_value=1, max_value=15, value=7, step=2)
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, value=25, step=5)
pop_size = st.sidebar.slider("Population size", min_value=10, max_value=50, value=20, step=5)
alpha_weight = st.sidebar.slider(
    "Accuracy weight (alpha)",
    min_value=0.50,
    max_value=0.99,
    value=0.90,
    step=0.05,
    help="Higher values prioritize accuracy; lower values place more weight on using fewer features.",
)
run_benchmark = st.sidebar.checkbox("Benchmark HBA against PSO and GA", value=True)

st.sidebar.markdown(
    "**Validation protocol**\n"
    "- Stratified 75/25 train-test split\n"
    "- Stratified cross-validation on training data\n"
    "- Median imputation and standardization fitted within each training fold"
)

st.subheader("1. Dataset")
data_source = st.radio(
    "Select data source",
    (
        "Diagnostic Breast Cancer Benchmark",
        "Upload custom CSV",
    ),
    horizontal=True,
)

x_frame = None
y_values = None
target_export = None
class_display_labels = None
feature_names = []
dataset_label = ""

if data_source == "Diagnostic Breast Cancer Benchmark":
    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer()
    x_frame = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y_values = breast_cancer.target.astype(int)
    target_export = pd.Series(
        breast_cancer.target_names[y_values],
        name="Target",
    )
    class_display_labels = [str(label) for label in breast_cancer.target_names]
    feature_names = list(x_frame.columns)
    dataset_label = "Diagnostic Breast Cancer Benchmark"
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV with numeric features and a classification target",
        type=["csv"],
    )
    if uploaded_file is not None:
        try:
            csv_frame = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()

        if csv_frame.empty or len(csv_frame.columns) == 0:
            st.error("The uploaded CSV has no rows or columns.")
            st.stop()

        target_column = st.selectbox(
            "Select target classification column",
            csv_frame.columns,
            index=len(csv_frame.columns) - 1,
        )
        rows_with_target = csv_frame.dropna(subset=[target_column])
        if rows_with_target.empty:
            st.error("The selected target column has no non-empty values.")
            st.stop()

        raw_features = rows_with_target.drop(columns=[target_column])
        x_frame = raw_features.select_dtypes(include=[np.number]).copy()
        ignored_columns = [column for column in raw_features.columns if column not in x_frame.columns]
        if ignored_columns:
            st.warning(
                "Only numeric feature columns are used. Ignored non-numeric columns: "
                + ", ".join(str(column) for column in ignored_columns)
            )
        if x_frame.shape[1] == 0:
            st.error("The CSV must contain at least one numeric feature column besides the target.")
            st.stop()

        target_export = rows_with_target[target_column].reset_index(drop=True)
        x_frame = x_frame.reset_index(drop=True)
        target_encoder = LabelEncoder()
        y_values = target_encoder.fit_transform(target_export.astype(str))
        class_display_labels = [str(label) for label in target_encoder.classes_]
        target_export.name = "Target"
        feature_names = list(x_frame.columns)
        dataset_label = uploaded_file.name

if x_frame is not None and y_values is not None:
    n_samples, n_features = x_frame.shape
    class_labels, class_counts = np.unique(y_values, return_counts=True)

    info_columns = st.columns(3)
    info_columns[0].info(f"**Samples:** {n_samples}")
    info_columns[1].info(f"**Numeric features:** {n_features}")
    info_columns[2].info(f"**Target classes:** {len(class_labels)}")

    with st.expander("Preview features"):
        st.dataframe(x_frame.head(10))

    if st.button("Run optimization", type="primary"):
        if len(class_labels) < 2:
            st.error("The target must contain at least two classes.")
            st.stop()
        if int(class_counts.min()) < 2:
            st.error("Each target class needs at least two samples for a stratified split.")
            st.stop()
        if len(class_labels) > 20:
            st.error(
                "The target has more than 20 classes. Select a categorical classification "
                "target rather than a continuous-value column."
            )
            st.stop()

        test_sample_count = math.ceil(0.25 * n_samples)
        train_sample_count = n_samples - test_sample_count
        if test_sample_count < len(class_labels) or train_sample_count < len(class_labels):
            st.error("There are too few samples for a stratified 75/25 split across all classes.")
            st.stop()

        x_train, x_test, y_train, y_test = train_test_split(
            x_frame.to_numpy(),
            y_values,
            test_size=0.25,
            random_state=42,
            stratify=y_values,
        )

        training_class_counts = np.bincount(y_train)
        cv_splits = min(5, int(training_class_counts.min()))
        if cv_splits < 2:
            st.error(
                "Each class needs at least two training samples for stratified cross-validation. "
                "Add more samples per class or choose another target."
            )
            st.stop()
        cv_validation_size = math.ceil(len(y_train) / cv_splits)
        minimum_cv_train_size = len(y_train) - cv_validation_size
        if k_neighbors > minimum_cv_train_size:
            st.error(
                f"KNN k ({k_neighbors}) cannot exceed the smallest cross-validation training "
                f"fold ({minimum_cv_train_size} samples)."
            )
            st.stop()

        st.caption(
            f"Using {cv_splits}-fold stratified cross-validation on the training split; "
            "the 25% holdout is reserved for final evaluation."
        )

        baseline_start = time.perf_counter()
        baseline_metrics, baseline_cm = evaluate_knn(
            x_train,
            x_test,
            y_train,
            y_test,
            k_neighbors,
            len(class_labels),
        )
        baseline_runtime = round(time.perf_counter() - baseline_start, 2)

        cross_validation = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42,
        )

        def objective_function(solution):
            selected = np.asarray(solution) > 0.5
            if not np.any(selected):
                return 1.0

            model = make_knn_pipeline(k_neighbors)
            cv_accuracy = cross_val_score(
                model,
                x_train[:, selected],
                y_train,
                cv=cross_validation,
                scoring="accuracy",
                error_score="raise",
            ).mean()
            feature_ratio = np.count_nonzero(selected) / n_features
            return alpha_weight * (1.0 - cv_accuracy) + (1.0 - alpha_weight) * feature_ratio

        problem = {
            "obj_func": objective_function,
            "bounds": FloatVar(
                lb=tuple(0.0 for _ in range(n_features)),
                ub=tuple(1.0 for _ in range(n_features)),
            ),
            "minmax": "min",
            "log_to": None,
        }

        optimizers = {
            "HBA": HBA.OriginalHBA(epoch=epochs, pop_size=pop_size),
        }
        if run_benchmark:
            optimizers.update(
                {
                    "PSO": PSO.OriginalPSO(epoch=epochs, pop_size=pop_size),
                    "GA": GA.BaseGA(epoch=epochs, pop_size=pop_size),
                }
            )

        results = {
            "Baseline KNN": {
                "metrics": baseline_metrics,
                "confusion_matrix": baseline_cm,
                "features": feature_names.copy(),
                "mask": np.ones(n_features, dtype=bool),
                "weights": None,
                "runtime": baseline_runtime,
                "best_fitness": None,
            }
        }
        convergence = {}
        progress = st.progress(0.0)

        for optimizer_index, (algorithm_name, optimizer) in enumerate(optimizers.items()):
            with st.spinner(
                f"Running {algorithm_name} ({epochs} epochs, population {pop_size})..."
            ):
                optimizer_start = time.perf_counter()
                best = optimizer.solve(problem)
                runtime = round(time.perf_counter() - optimizer_start, 2)

            weights = np.asarray(best.solution)
            selected_mask = weights > 0.5
            if not np.any(selected_mask):
                selected_mask[np.argmax(weights)] = True

            metrics, matrix = evaluate_knn(
                x_train[:, selected_mask],
                x_test[:, selected_mask],
                y_train,
                y_test,
                k_neighbors,
                len(class_labels),
            )
            results[algorithm_name] = {
                "metrics": metrics,
                "confusion_matrix": matrix,
                "features": list(np.asarray(feature_names)[selected_mask]),
                "mask": selected_mask,
                "weights": weights,
                "runtime": runtime,
                "best_fitness": float(best.target.fitness),
            }
            convergence[algorithm_name] = list(optimizer.history.list_global_best_fit)
            progress.progress((optimizer_index + 1) / len(optimizers))

        hba_result = results["HBA"]

        tab_comparison, tab_convergence, tab_weights, tab_exports = st.tabs(
            (
                "Benchmark Comparison",
                "Multi-Algorithm Convergence Curve",
                "HBA Feature Weights",
                "CSV Exports",
            )
        )

        with tab_comparison:
            st.subheader("HBA holdout metrics")
            metric_columns = st.columns(5)
            for index, metric_name in enumerate(
                ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")
            ):
                metric_value = hba_result["metrics"][metric_name]
                metric_columns[index].metric(metric_name, f"{metric_value:.2%}")

            comparison_rows = []
            for algorithm_name, result in results.items():
                comparison_rows.append(
                    {
                        "Algorithm": algorithm_name,
                        **{
                            name: f"{value:.2%}"
                            for name, value in result["metrics"].items()
                        },
                        "Features selected": len(result["features"]),
                        "Runtime (seconds)": result["runtime"],
                    }
                )
            st.subheader("Algorithm comparison")
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True)

            st.subheader("Confusion matrices")
            for algorithm_name, result in results.items():
                with st.expander(f"{algorithm_name} confusion matrix"):
                    matrix = result["confusion_matrix"]
                    matrix_frame = pd.DataFrame(
                        matrix,
                        index=[f"Actual {label}" for label in class_display_labels],
                        columns=[f"Predicted {label}" for label in class_display_labels],
                    )
                    st.dataframe(matrix_frame)

        with tab_convergence:
            st.subheader("Best fitness by epoch")
            convergence_frame = pd.DataFrame(
                {
                    algorithm_name: pd.Series(
                        values,
                        index=range(1, len(values) + 1),
                    )
                    for algorithm_name, values in convergence.items()
                }
            )
            convergence_frame.index.name = "Epoch"
            st.line_chart(convergence_frame)

        with tab_weights:
            st.subheader(
                f"HBA feature weights ({len(hba_result['features'])} of {n_features} selected)"
            )
            weights_frame = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Continuous weight": hba_result["weights"],
                    "Selected": hba_result["mask"],
                }
            ).sort_values("Continuous weight", ascending=False)
            st.dataframe(weights_frame, hide_index=True)

        with tab_exports:
            st.subheader("Download research artifacts")
            optimized_frame = x_frame.loc[:, hba_result["features"]].copy()
            optimized_frame.insert(len(optimized_frame.columns), "Target", target_export)
            st.download_button(
                "Download optimized feature dataset (CSV)",
                data=optimized_frame.to_csv(index=False).encode("utf-8"),
                file_name="hba_knn_optimized_dataset.csv",
                mime="text/csv",
            )

            report_rows = []
            for algorithm_name, result in results.items():
                report_rows.append(
                    {
                        "Dataset": dataset_label,
                        "Algorithm": algorithm_name,
                        "KNN_k": k_neighbors,
                        "Epochs": epochs,
                        "Population_size": pop_size,
                        "Accuracy_weight": alpha_weight,
                        "Original_features": n_features,
                        "Selected_features": len(result["features"]),
                        "Accuracy": result["metrics"]["Accuracy"],
                        "Precision": result["metrics"]["Precision"],
                        "Recall": result["metrics"]["Recall"],
                        "F1_score": result["metrics"]["F1-Score"],
                        "ROC_AUC": result["metrics"]["ROC-AUC"],
                        "Runtime_seconds": result["runtime"],
                        "Best_fitness": result["best_fitness"],
                    }
                )
            report_frame = pd.DataFrame(report_rows)
            st.download_button(
                "Download benchmark report (CSV)",
                data=report_frame.to_csv(index=False).encode("utf-8"),
                file_name="hba_knn_benchmark_report.csv",
                mime="text/csv",
            )

st.markdown("---")
st.markdown(
    "**Built by Ukasha M. I. Kirfi** | *Computer Science Alumnus, Federal University of Kashere (FUK)*\n\n"
    "📚 **Research inspiration:** Honey Badger Optimization survey by "
    "*Hassan, Abdullahi, Isuwa et al. (2024)* and the 2026 application by "
    "*Abed et al.* in *AIP Advances*, combining HBA with K-Nearest Neighbors (KNN) "
    "for tumor classification."
)
