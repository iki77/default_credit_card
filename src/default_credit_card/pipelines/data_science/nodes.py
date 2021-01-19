# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import math
import datetime
from collections import Counter
from timeit import default_timer as timer

# Feature engineering packages
from dask.distributed import Client, LocalCluster
import featuretools as ft

# Preprocessing packages
from category_encoders import CatBoostEncoder, JamesSteinEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2, f_classif, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Sampling packages
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    RandomUnderSampler,
    TomekLinks,
)
from imblearn.pipeline import Pipeline as PipelineImb

# Modelling packages
from sklearn.ensemble import (
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Calibarion packages
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Evaluation packages
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    plot_roc_curve,
    roc_auc_score,
)


def _get_time_delta(seconds=float) -> str:
    return str(datetime.timedelta(seconds=seconds))


def _downcast_numeric(x):
    if x.dtype == np.float64:
        return x.astype(np.float32)
    elif x.dtype == np.int64:
        return x.astype(np.int32)
    return x


def _create_client_entityset(
    clients: pd.DataFrame, profiles: pd.DataFrame, entity_id: str
) -> ft.EntitySet:
    es = ft.EntitySet(id=entity_id)
    es = es.entity_from_dataframe(
        entity_id="clients",
        dataframe=clients,
        index="ID",
        variable_types={
            "MARRIAGE": ft.variable_types.Categorical,
            "SEX": ft.variable_types.Categorical,
            "EDUCATION": ft.variable_types.Ordinal,
        },
    )
    es = es.entity_from_dataframe(
        entity_id="profiles",
        dataframe=profiles,
        index="PROFILE_ID",
        time_index="MONTH",
        make_index=True,
        variable_types={
            "ID": ft.variable_types.Id,
            "CREDIT_USE": ft.variable_types.Boolean,
        },
    )
    es = es.add_relationship(ft.Relationship(es["clients"]["ID"], es["profiles"]["ID"]))

    return es


def _initialize_dask_client(n_workers: int = 2, threads: Optional[int] = None) -> List:
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads)

    return [Client(cluster), cluster]


def _get_column_dtype(df: pd.DataFrame) -> Dict:
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="boolean").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Add ID cols if it exists
    id_cols: List = []
    for col in ["ID"]:
        if col not in all_cols:
            continue
        if col in cat_cols:
            cat_cols.remove(col)
        if col in num_cols:
            num_cols.remove(col)
        id_cols.append(col)

    # Add default category cols if it exists but not detected as category
    for col in [
        "MARRIAGE",
        "SEX",
    ]:
        if col not in all_cols:
            continue
        if col in num_cols:
            num_cols.remove(col)
        if col not in cat_cols:
            cat_cols.append(col)
    cat_cols.sort()

    # Add default ordinal category if it exists
    ordi_cols: List = []
    for col in ["EDUCATION"]:
        if col not in all_cols:
            continue
        if col in cat_cols:
            cat_cols.remove(col)
        if col in num_cols:
            num_cols.remove(col)
        ordi_cols.append(col)
    ordi_cols.sort()

    # Add boolean cols if it exists but not detected as boolean
    for col in num_cols:
        if set(df[col].astype(float).unique()) == {0, 1}:
            bool_cols.append(col)
    bool_cols.sort()
    for col in bool_cols:
        if col in num_cols:
            num_cols.remove(col)
    num_cols.sort()

    # Seperate numerical columns with skewed distribution
    skew_check = pd.DataFrame(df[num_cols].skew(), columns=["skew"])

    num_skew_cols = skew_check[np.abs(skew_check["skew"]) >= 1].index.tolist()
    num_skew_cols.sort()
    for col in num_skew_cols:
        num_cols.remove(col)

    col_dict = {
        "id": id_cols,
        "num_normal": num_cols,
        "num_skewed": num_skew_cols,
        "ordinal": ordi_cols,
        "boolean": bool_cols,
        "category": cat_cols,
    }

    return col_dict


def _enforce_dtype(df: pd.DataFrame) -> pd.DataFrame:
    col_dict = _get_column_dtype(df)

    # Check if ID exists
    if col_dict["id"]:
        df[col_dict["id"]] = df[col_dict["id"]].astype("object")

    # Enforce dtype
    df[col_dict["num_normal"]] = df[col_dict["num_normal"]].apply(_downcast_numeric)
    df[col_dict["num_skewed"]] = df[col_dict["num_skewed"]].apply(_downcast_numeric)
    df[col_dict["boolean"]] = df[col_dict["boolean"]].astype(bool)
    df[col_dict["ordinal"]] = df[col_dict["ordinal"]].astype("category")
    df[col_dict["category"]] = df[col_dict["category"]].astype("category")

    return df


def _get_ct_feature_names(ct: ColumnTransformer) -> List:
    feature_names = []
    for name, trans, column in ct.transformers_:
        if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
            continue
        if trans == "passthrough":
            feature_names.extend(column)
            continue
        # if hasattr(trans, "get_feature_names"):
        #     feature_names.extend(trans.get_feature_names(column))
        #     continue

        feature_names.extend(column)

    return feature_names


def _get_ct_support(ct: ColumnTransformer) -> List:
    support_list = []
    for name, trans, column in ct.transformers_:
        if not hasattr(trans, "get_support"):
            continue
        support_list.extend(trans.get_support())
    return support_list


def _inverse_ct_transform(df: pd.DataFrame, ct: ColumnTransformer) -> pd.DataFrame:
    df_inverse = df.copy()
    for name, trans, column in ct.transformers_:
        if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
            continue
        if trans == "passthrough":
            continue
        if hasattr(trans, "inverse_transform"):
            df_inverse[column] = trans.inverse_transform(df_inverse[column])
            continue

    return df_inverse


def _get_oversample_strategy(series: pd.Series, multiplier: float = 1.0) -> Dict:
    if multiplier <= 0:
        raise ValueError("Multiplier must be within greater than 0.")

    counter: Counter = Counter(series)

    # Store the median sample of all labels
    median_sample = np.median([sample for label, sample in counter.items()])
    recommended_sample = math.ceil(median_sample * multiplier)

    sampling_strat: Dict = {}
    # Populate sampling stategy for oversampling
    for label, sample in counter.most_common():
        if sample <= median_sample:
            # Oversample label if its sample lower than median sample
            sampling_strat[label] = recommended_sample
            continue

        sampling_strat[label] = sample

    return sampling_strat


def _remove_unused_transformers(transformers: List) -> List:
    used_trans = transformers
    for i, trans_set in enumerate(used_trans):
        name, trans, column = trans_set
        if not column:
            used_trans.pop(i)

    return used_trans


def _remove_unused_steps(steps: List, remove_clf: bool = False) -> List:
    used_steps = steps
    for i, step_set in enumerate(used_steps):
        name, trans = step_set
        if not trans:
            used_steps.pop(i)
            continue
        if remove_clf:
            if name == "classifier":
                used_steps.pop(i)

    return used_steps


def add_feature_profiles(profiles: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """Add additional features to profiles

    Args:
        profiles: Data of client monthly profiles.
        clients: Data of normalized client.
    Returns:
        Monthly profile with additional features
    """
    profiles = pd.merge(profiles, clients[["ID", "LIMIT_BAL"]], how="left", on="ID")

    # Determine the percentage threshold of used balance from the limit
    profiles["LIMIT_THRES"] = profiles["BILL_AMT"] / profiles["LIMIT_BAL"]

    # Determine if the client use credit card on that month
    profiles["CREDIT_USE"] = np.where(
        (profiles["PAY_STATUS"] == 0)
        & (profiles["BILL_AMT"] == 0)
        & (profiles["PAY_AMT"] == 0),
        False,
        True,
    )

    profiles.drop(columns="LIMIT_BAL", inplace=True)

    return profiles


def split_data(
    clients: pd.DataFrame,
    profiles: pd.DataFrame,
    labels: pd.DataFrame,
    parameters: Dict,
) -> List[pd.DataFrame]:
    """Splits data into training, calibration and test sets.

    Args:
        clients: Data of normalized client.
        profiles: Data of client monthly profiles.
        labels: Data of next month payment default status.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing split data.
    """
    (clients_train, clients_test, labels_train, labels_test,) = train_test_split(
        clients,
        labels,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=labels["DEFAULT_PAY"],
    )

    profiles_train = profiles[profiles["ID"].isin(labels_train["ID"])]
    profiles_test = profiles[profiles["ID"].isin(labels_test["ID"])]

    return [
        clients_train,
        profiles_train,
        labels_train,
        clients_test,
        profiles_test,
        labels_test,
    ]


def create_feature_definitions(
    clients_train: pd.DataFrame, profiles_train: pd.DataFrame, parameters: Dict
) -> List:
    """Create feature definitions and features set using DFS.

    Args:
        clients_train: Training data of normalized client.
        profiles_train: Training data of client monthly profiles.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing calculated features and its feature definitions from DFS.
    """
    # Store client columns
    client_cols = clients_train.drop(columns="ID").columns.tolist()

    # Initialize dask client
    dask_client, dask_cluster = _initialize_dask_client(n_workers=2)

    # Log original features
    logger = logging.getLogger(__name__)
    logger.info(
        f"Original features, excluding ID and MONTH from data sources: {(clients_train.shape[1] - 1) + (profiles_train.shape[1] - 2)} ."
    )

    # Create the EntitySet
    es = _create_client_entityset(clients_train, profiles_train, "client_train_set")

    # Create seed features
    retirement_age = ft.Feature(es["clients"]["AGE"]) >= 55

    # Start DFS
    features, feature_defs = ft.dfs(
        entityset=es,
        target_entity="clients",
        agg_primitives=[
            "all",
            "count",
            "last",
            "max",
            "mean",
            "median",
            "min",
            "num_true",
            "percent_true",
            "std",
            "skew",
            "sum",
            "trend",
        ],
        trans_primitives=["percentile"],
        seed_features=[retirement_age],
        max_depth=2,
        dask_kwargs={"cluster": dask_cluster},
        verbose=True,
    )

    dask_client.close()

    # Log features created after DFS
    logger.info(f"Features after DFS: {len(feature_defs)} features.")

    # Remove highly null features
    features, feature_defs = ft.selection.remove_highly_null_features(
        features,
        features=feature_defs,
        pct_null_threshold=parameters["null_thres"],
    )
    logger.info(f"Features after removing highly null features: {len(feature_defs)}")

    # Remove single value features
    features, feature_defs = ft.selection.remove_single_value_features(
        features, features=feature_defs
    )
    logger.info(f"Features after removing single value features: {len(feature_defs)}")

    # Remove highly correlated features
    features, feature_defs = ft.selection.remove_highly_correlated_features(
        features,
        features=feature_defs,
        pct_corr_threshold=parameters["corr_thres"],
        features_to_keep=client_cols,
    )
    logger.info(
        f"Final features after removing highly correlated features: {len(feature_defs)}"
    )

    # Reindex based on ID of clients
    features = features.reindex(index=clients_train["ID"])
    features.reset_index(inplace=True)

    # Enforce dtype
    features = _enforce_dtype(features)

    # Make sure feature matrix have the same index as clients
    features.index = clients_train.index

    return [features, feature_defs]


def calculate_features(
    feature_defs: List, clients: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Calculate features from existing feature definitions.

    Args:
        feature_defs: Feature definitions from DFS.
        clients: Data of normalized client.
        profiles: Data of client monthly profiles.
    Returns:
        Independent features calculated from feature definitions.
    """
    # Initialize dask client
    dask_client, dask_cluster = _initialize_dask_client(n_workers=2)

    # Create the EntitySet
    es = _create_client_entityset(clients, profiles, "client_other_set")

    # Calculate feature matrix
    features = ft.calculate_feature_matrix(
        features=feature_defs,
        entityset=es,
        dask_kwargs={"cluster": dask_cluster},
        verbose=True,
    )

    dask_client.close()

    # Reindex based on ID of clients
    features = features.reindex(index=clients["ID"])
    features.reset_index(inplace=True)

    # Enforce dtype
    features = _enforce_dtype(features)

    # Make sure feature matrix have the same index as clients
    features.index = clients.index

    return features


def train_imputer(features: pd.DataFrame) -> Pipeline:
    """Train imputer.

    Args:
        features_train: Training data of independent features.
    Returns:
        Trained imputer.
    """
    col_dict = _get_column_dtype(features)

    # Create transformers for each dtype
    transformers = [
        ("num_n_imputer", SimpleImputer(strategy="median"), col_dict["num_normal"]),
        ("num_s_imputer", SimpleImputer(strategy="median"), col_dict["num_skewed"]),
        (
            "ordi_imputer",
            SimpleImputer(strategy="constant", fill_value=0),
            col_dict["ordinal"],
        ),
        ("bool_pass", "passthrough", col_dict["boolean"]),
        (
            "cat_imputer",
            SimpleImputer(strategy="constant", fill_value=0),
            col_dict["category"],
        ),
    ]
    transformers = _remove_unused_transformers(transformers)

    # Combine the transformers as imputer
    imputer = ColumnTransformer(transformers=transformers)
    imputer.fit(features)

    return imputer


def impute_missing_values(
    imputer: ColumnTransformer, features: pd.DataFrame
) -> pd.DataFrame:
    """Impute features using trained imputer.

    Args:
        features: Data of independent features.
        imputer: Trained imputer.
    Returns:
        Imputed features using the trained imputer.
    """
    # Remap imputer output to DataFrame
    input_cols = _get_ct_feature_names(imputer)
    features_imp = pd.DataFrame(imputer.transform(features), columns=input_cols)

    # Reindex based on ID of clients
    features_imp.index = features["ID"]
    features_imp = features_imp.reindex(index=features["ID"])
    features_imp.reset_index(inplace=True)

    # Enforce dtype
    features_imp = _enforce_dtype(features_imp)

    # Make sure feature matrix have the same index as clients
    features_imp.index = features.index

    return features_imp


def train_outlier_detector(
    features_train: pd.DataFrame, labels_train: pd.DataFrame, parameters: Dict
) -> Pipeline:
    """Train oulier detector and remove the outliers from features and its labels.

    Args:
        features_train: Training data of independent features.
        labels_train: Training data of next month payment default status.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing features, its default labels without outliers and its trained outlier detector
    """
    col_dict = _get_column_dtype(features_train)

    if labels_train.shape[0] == features_train.shape[0]:
        labels_train.index = features_train.index

    # Create transformers for each dtype
    transformers = [
        ("num_n_trans", StandardScaler(), col_dict["num_normal"]),
        (
            "num_s_trans",
            QuantileTransformer(random_state=parameters["random_state"]),
            col_dict["num_skewed"],
        ),
        ("ordi_trans", "passthrough", col_dict["ordinal"]),
        ("bool_pass", "passthrough", col_dict["boolean"]),
        (
            "cat_trans",
            CatBoostEncoder(random_state=parameters["random_state"], return_df=False),
            col_dict["category"],
        ),
    ]
    transformers = _remove_unused_transformers(transformers)

    # Log original features, excluding ID
    logger = logging.getLogger(__name__)
    features_train.replace([np.inf, -np.inf], np.nan)
    logger.info(features_train.columns[features_train.isna().any()])

    # Combine the transformers as preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    #  Extract target
    target_train = labels_train["DEFAULT_PAY"]

    # Create outlier detector pipeline and train it
    detector = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("detector", IsolationForest(n_jobs=-1)),
        ]
    )
    detector.fit(features_train, target_train)

    return detector


def remove_outliers(
    detector: Pipeline,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> List:
    """Remove outliers from features and its labels using trained outlier detector.

    Args:
        detector: Trained outlier detector.
        features: Data of independent features.
        labels: Data of next month payment default status.
    Returns:
        A list containing features and its default labels without outliers.
    """

    if labels.shape[0] == features.shape[0]:
        labels.index = features.index

    # Log original rows
    logger = logging.getLogger(__name__)
    logger.info("Original rows: {}".format(features.shape[0]))

    # Store predicted outlier labels
    features["OUTLIER"] = detector.predict(features)

    # Remove outliers (outlier = -1)
    features = features[features["OUTLIER"] != -1]
    labels = labels[labels["ID"].isin(features["ID"])]
    features.drop(columns="OUTLIER", inplace=True)

    logger.info("Final rows after removing outliers: {}".format(features.shape[0]))

    # Enforce dtype
    features = _enforce_dtype(features)

    return [features, labels]


def train_feature_selector(
    features_train: pd.DataFrame,
    labels_train: pd.DataFrame,
    parameters: Dict,
) -> Pipeline:
    """Train feature selector and select only relevant features fot the label.

    Args:
        features_train: Training data of independent features.
        labels_train: Training data of next month payment default status.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing relevant features, and the trained feature selector
    """
    col_dict = _get_column_dtype(features_train)

    if labels_train.shape[0] == features_train.shape[0]:
        labels_train.index = features_train.index

    # Create transformers for each dtype
    transformers = [
        ("num_n_trans", StandardScaler(), col_dict["num_normal"]),
        (
            "num_s_trans",
            QuantileTransformer(random_state=parameters["random_state"]),
            col_dict["num_skewed"],
        ),
        ("ordi_trans", "passthrough", col_dict["ordinal"]),
        ("bool_pass", "passthrough", col_dict["boolean"]),
        (
            "cat_trans",
            JamesSteinEncoder(random_state=parameters["random_state"], return_df=False),
            col_dict["category"],
        ),
    ]
    transformers = _remove_unused_transformers(transformers)

    # Combine the transformers as preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    num_cols = col_dict["num_normal"] + col_dict["num_skewed"]
    nomi_cols = col_dict["ordinal"] + col_dict["boolean"] + col_dict["category"]

    selector_ct = ColumnTransformer(
        transformers=[
            (
                "num_selector",
                SelectPercentile(f_classif, percentile=parameters["numeric_pct"]),
                [x for x in range(0, len(num_cols))],
            ),
            (
                "nomi_selector",
                SelectPercentile(chi2, percentile=parameters["nominal_pct"]),
                [x for x in range(len(num_cols), len(num_cols) + len(nomi_cols))],
            ),
        ]
    )

    #  Extract target
    target_train = labels_train["DEFAULT_PAY"]

    # Create feature selector pipeline and train it
    selector = Pipeline(
        steps=[("preprocessor", preprocessor), ("selector", selector_ct)]
    )
    selector.fit(features_train, target_train)

    return selector


def select_relevant_features(
    selector: Pipeline, features: pd.DataFrame
) -> pd.DataFrame:
    """Select relevant features using trained feature selector

    Args:
        selector: Trained feature selector.
        eatures: Data of independent features.
    Returns:
        Relevant features selected using the trained feature selector.
    """
    # Log original features, excluding ID
    logger = logging.getLogger(__name__)
    logger.info("Original features: {}".format(features.shape[1] - 1))

    # Remap feature selector output to DataFrame
    input_cols = _get_ct_feature_names(selector.named_steps["preprocessor"])
    selected_cols = _get_ct_support(selector.named_steps["selector"])

    # Filter features that are not selected
    features_sel = features[input_cols]
    features_sel = features_sel.iloc[:, selected_cols]

    logger.info("Final features after selection: {}".format(features_sel.shape[1]))

    # Reindex based on ID of clients
    features_sel.index = features["ID"]
    features_sel = features_sel.reindex(index=features["ID"])
    features_sel.reset_index(inplace=True)

    # Enforce dtype
    features_sel = _enforce_dtype(features_sel)

    # Make sure feature matrix have the same index as clients
    features_sel.index = features.index

    return features_sel


def find_best_resampler(
    features_train: pd.DataFrame, labels_train: pd.DataFrame, parameters: Dict
) -> List:
    """Compare several resamplers and find the best one to handle imbalanced labels.

    Args:
        features_train: Training data of independent features.
        labels_train: Training data of next month payment default status.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing the best resampler and the search CV results as DataFrame.
    """
    col_dict = _get_column_dtype(features_train)

    if labels_train.shape[0] == features_train.shape[0]:
        labels_train.index = features_train.index

    # Create transformers for each dtype
    transformers = [
        ("num_n_trans", StandardScaler(), col_dict["num_normal"]),
        (
            "num_s_trans",
            QuantileTransformer(random_state=parameters["random_state"]),
            col_dict["num_skewed"],
        ),
        ("ordi_trans", "passthrough", col_dict["ordinal"]),
        ("bool_pass", "passthrough", col_dict["boolean"]),
        (
            "cat_trans",
            JamesSteinEncoder(random_state=parameters["random_state"], return_df=False),
            col_dict["category"],
        ),
    ]
    transformers = _remove_unused_transformers(transformers)

    # Combine the transformers as preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    num_cols = col_dict["num_normal"] + col_dict["num_skewed"]
    nomi_cols = col_dict["ordinal"] + col_dict["boolean"] + col_dict["category"]

    #  Extract target
    target_train = labels_train["DEFAULT_PAY"]

    # Initalize samplers
    smotenc_smpl = SMOTENC(
        categorical_features=[
            x for x in range(len(num_cols), len(num_cols) + len(nomi_cols))
        ],
        n_jobs=-1,
    )
    ro_smpl = RandomOverSampler()
    enn_smpl = EditedNearestNeighbours(n_jobs=-1)
    tl_smpl = TomekLinks(n_jobs=-1)
    ru_smpl = RandomUnderSampler()

    # Initalize classifier
    clf = ExtraTreesClassifier(max_depth=10, n_jobs=-1)

    # Create parameter grid
    param_grid = {
        "sampler": [None, smotenc_smpl, ro_smpl, enn_smpl, tl_smpl, ru_smpl],
        "classifier": [clf],
    }

    # Create classifier pipeline
    resampler = PipelineImb(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", smotenc_smpl),
            ("classifier", clf),
        ]
    )

    # Start grid search
    search_cv = GridSearchCV(
        resampler,
        param_grid=param_grid,
        scoring=[
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        refit="f1",
        error_score=0,
        verbose=2,
    )

    timer_start = timer()
    search_cv.fit(features_train, target_train)
    timer_end = timer()

    # Log search duration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Best resampler search elapsed time : {_get_time_delta(timer_end - timer_start)}."
    )

    # Save search result as DataFrame
    search_results = pd.DataFrame(search_cv.cv_results_).sort_values(
        by=["rank_test_f1"]
    )

    # Remove unused steps from resampler
    resampler = search_cv.best_estimator_
    resampler.set_params(
        steps=_remove_unused_steps(steps=resampler.steps, remove_clf=True)
    )

    return [resampler, search_results]


def resample_data(
    resampler: Pipeline, features: pd.DataFrame, labels: pd.DataFrame
) -> List:
    """Resample data using trained resampler.

    Args:
        resampler: Trained resampler.
        features: Data of independent features.
        labels: Data of next month payment default status.
    Returns:
        A list containing the resampled features and its labels.
    """
    if labels.shape[0] == features.shape[0]:
        labels.index = features.index

    features_res = features
    labels_res = labels

    if "sampler" in resampler.named_steps:
        #  Extract target
        target = labels["DEFAULT_PAY"]

        features_res, target_res = resampler.fit_resample(features, target)

        # Remap resampler output to DataFrame
        input_cols = _get_ct_feature_names(resampler.named_steps["preprocessor"])

        features_res = pd.DataFrame(features_res, columns=input_cols)
        features_res = _inverse_ct_transform(
            df=features_res, ct=resampler.named_steps["preprocessor"]
        )
        labels_res = pd.DataFrame(target_res, columns=["DEFAULT_PAY"])

        # Create fake ID from the index of resampled data
        features_res.index.name = "ID"
        labels_res.index.name = "ID"
        features_res.reset_index(inplace=True)
        labels_res.reset_index(inplace=True)

        # Enforce dtype
        features_res = _enforce_dtype(features_res)
        labels_res["ID"] = labels_res["ID"].astype("object")
        labels_res["DEFAULT_PAY"] = labels_res["DEFAULT_PAY"].astype(np.float32)

        # Make sure features have the same index as labels
        features_res.index = labels_res.index

    return [features_res, labels_res]


def find_best_classifier(
    features_train: pd.DataFrame, labels_train: pd.DataFrame, parameters: Dict
) -> List:
    """Compare several classifiers and find the best one to handle imbalanced labels.

    Args:
        features_train: Training data of independent features.
        labels_train: Training data of next month payment default status.
        parameters: Parameters defined in parameters.yml.
    Returns:
        A list containing the best classifier and the search CV results as DataFrame.
    """
    col_dict = _get_column_dtype(features_train)

    if labels_train.shape[0] == features_train.shape[0]:
        labels_train.index = features_train.index

    # Create transformers for each dtype
    transformers = [
        ("num_n_trans", StandardScaler(), col_dict["num_normal"]),
        (
            "num_s_trans",
            QuantileTransformer(random_state=parameters["random_state"]),
            col_dict["num_skewed"],
        ),
        ("ordi_trans", "passthrough", col_dict["ordinal"]),
        ("bool_pass", "passthrough", col_dict["boolean"]),
        (
            "cat_trans",
            CatBoostEncoder(random_state=parameters["random_state"], return_df=False),
            col_dict["category"],
        ),
    ]
    transformers = _remove_unused_transformers(transformers)

    # Combine the transformers as preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    num_cols = col_dict["num_normal"] + col_dict["num_skewed"]
    nomi_cols = col_dict["ordinal"] + col_dict["boolean"] + col_dict["category"]

    #  Extract target
    target_train = labels_train["DEFAULT_PAY"]

    # Initalize classifiers
    gnb_clf = GaussianNB()
    lr_clf = LogisticRegression(max_iter=200, n_jobs=-1)
    knn_clf = KNeighborsClassifier(n_jobs=-1)
    xt_clf = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
    rf_clf = RandomForestClassifier(max_depth=10, n_jobs=-1)
    mlp_clf = MLPClassifier(max_iter=200, early_stopping=True)
    lgbm_clf = LGBMClassifier(max_depth=10, num_leaves=500, n_jobs=-1)

    # Create parameter grid
    param_grid = {
        "classifier": [
            gnb_clf,
            lr_clf,
            knn_clf,
            xt_clf,
            rf_clf,
            mlp_clf,
            lgbm_clf,
        ],
    }

    # Create classifier pipeline
    classifier = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", lr_clf)]
    )

    # Start grid search
    search_cv = GridSearchCV(
        classifier,
        param_grid=param_grid,
        scoring=[
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        refit="f1",
        error_score=0,
        verbose=2,
    )

    timer_start = timer()
    search_cv.fit(features_train, target_train)
    timer_end = timer()

    # Log search duration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Best classifier search elapsed time : {_get_time_delta(timer_end - timer_start)}."
    )

    # Save search result as DataFrame
    search_results = pd.DataFrame(search_cv.cv_results_).sort_values(
        by=["rank_test_f1"]
    )

    classifier = search_cv.best_estimator_

    return [classifier, search_results]


def evaluate_classifier(
    classifier: Pipeline, features_test: np.ndarray, labels_test: np.ndarray
):
    """Calculate the coefficient of determination and log the result.

    Args:
        classifier: Trained classifier.
        features_test: Testing data of independent features.
        labels_test: Testing data of next month payment default status.

    """
    target_test = labels_test["DEFAULT_PAY"]
    target_pred = classifier.predict(features_test)
    score = f1_score(target_test, target_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Classifier has a coefficient F1 of {score:3f}.")


# def split_data(data: pd.DataFrame, parameters: Dict) -> List:
#     """Splits data into training and test sets.

#     Args:
#         data: Source data.
#         parameters: Parameters defined in parameters.yml.
#     Returns:
#         A list containing split data.

#     """
#     X = data[
#         [
#             "engines",
#             "passenger_capacity",
#             "crew",
#             "d_check_complete",
#             "moon_clearance_complete",
#         ]
#     ].values
#     y = data["price"].values
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
#     )

#     return [X_train, X_test, y_train, y_test]


# def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
#     """Train the linear regression model.

#     Args:
#         X_train: Training data of independent features.
#         y_train: Training data for price.

#     Returns:
#         Trained model.

#     """
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)
#     return regressor


# def evaluate_model(regressor: LinearRegression, X_test: np.ndarray, y_test: np.ndarray):
#     """Calculate the coefficient of determination and log the result.

#     Args:
#         regressor: Trained model.
#         X_test: Testing data of independent features.
#         y_test: Testing data for price.

#     """
#     y_pred = regressor.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has a coefficient R^2 of %.3f.", score)
