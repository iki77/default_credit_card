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
from kedro.pipeline import Pipeline, node
from numpy.lib.function_base import select

from .nodes import (
    add_feature_profiles,
    calculate_features,
    create_feature_definitions,
    evaluate_classifier,
    find_best_classifier,
    find_best_resampler,
    remove_outliers,
    resample_data,
    select_relevant_features,
    train_feature_selector,
    train_imputer,
    impute_missing_values,
    train_outlier_detector,
    split_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=add_feature_profiles,
                inputs=["profiles", "normalized_clients"],
                outputs="added_profiles",
                name="adding_features_to_profiles",
            ),
            node(
                func=split_data,
                inputs=[
                    "normalized_clients",
                    "added_profiles",
                    "labels",
                    "parameters",
                ],
                outputs=[
                    "clients_train",
                    "profiles_train",
                    "labels_train",
                    "clients_test",
                    "profiles_test",
                    "labels_test",
                ],
            ),
            node(
                func=create_feature_definitions,
                inputs=["clients_train", "profiles_train", "parameters"],
                outputs=["features_train", "feature_defs"],
                name="creating_feature_definitions",
            ),
            node(
                func=calculate_features,
                inputs=["feature_defs", "clients_test", "profiles_test"],
                outputs="features_test",
                name="calculating_features_on_test_data",
            ),
            node(
                func=train_imputer,
                inputs="features_train",
                outputs="imputer",
                name="training_imputer",
            ),
            node(
                func=impute_missing_values,
                inputs=["imputer", "features_train"],
                outputs="imputed_features_train",
                name="imputing_training_data",
            ),
            node(
                func=impute_missing_values,
                inputs=["imputer", "features_test"],
                outputs="imputed_features_test",
                name="imputing_test_data",
            ),
            node(
                func=train_outlier_detector,
                inputs=["imputed_features_train", "labels_train", "parameters"],
                outputs="outlier_detector",
                name="training_outlier_detector",
            ),
            node(
                func=remove_outliers,
                inputs=["outlier_detector", "imputed_features_train", "labels_train"],
                outputs=[
                    "inliner_features_train",
                    "inliner_labels_train",
                ],
                name="remove_outlier_from_training_data",
            ),
            node(
                func=train_feature_selector,
                inputs=[
                    "inliner_features_train",
                    "inliner_labels_train",
                    "parameters",
                ],
                outputs="feature_selector",
                name="training_feature_selector",
            ),
            node(
                func=select_relevant_features,
                inputs=["feature_selector", "inliner_features_train"],
                outputs="selected_features_train",
                name="selecting_features_on_calibration_set",
            ),
            node(
                func=select_relevant_features,
                inputs=["feature_selector", "imputed_features_test"],
                outputs="selected_features_test",
                name="selecting_features_on_test_data",
            ),
            node(
                func=find_best_resampler,
                inputs=[
                    "selected_features_train",
                    "inliner_labels_train",
                    "parameters",
                ],
                outputs=["resampler", "resampler_search_results"],
                name="finding_best_resampler",
            ),
            node(
                func=resample_data,
                inputs=["resampler", "selected_features_train", "inliner_labels_train"],
                outputs=["resampled_features_train", "resampled_labels_train"],
                name="resampling_training_data",
            ),
            node(
                func=find_best_classifier,
                inputs=[
                    "resampled_features_train",
                    "resampled_labels_train",
                    "parameters",
                ],
                outputs=["classifier", "classifier_search_results"],
                name="finding_best_classifier",
            ),
            node(
                func=evaluate_classifier,
                inputs=["classifier", "selected_features_test", "labels_test"],
                outputs=None,
                name="evaluating_classifier",
            ),
        ]
    )


# def create_pipeline(**kwargs):
#     return Pipeline(
#         [
#             node(
#                 func=split_data,
#                 inputs=["master_table", "parameters"],
#                 outputs=["X_train", "X_test", "y_train", "y_test"],
#             ),
#             node(func=train_model, inputs=["X_train", "y_train"], outputs="regressor"),
#             node(
#                 func=evaluate_model,
#                 inputs=["regressor", "X_test", "y_test"],
#                 outputs=None,
#             ),
#         ]
#     )
