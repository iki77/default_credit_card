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

from typing import Dict, List


import numpy as np
import pandas as pd


def _downcast_numeric(x):
    if x.dtype == np.float64:
        return x.astype(np.float32)
    elif x.dtype == np.int64:
        return x.astype(np.int32)
    return x


def _enforce_gender_format(x):
    if x not in [1, 2, np.nan]:
        return 1
    return x


def _enforce_education_format(x):
    if x not in [1, 2, 3, 4, np.nan]:
        return 4
    return x


def _enforce_marital_format(x):
    if x not in [1, 2, 3, np.nan]:
        return 3
    return x


def _enforce_repayment_status(x):
    if x <= 0:
        return 0
    return x


def _melt_client_profile(
    clients: pd.DataFrame, var_name: str, value_name: str
) -> pd.DataFrame:
    return pd.melt(
        clients,
        id_vars="ID",
        value_vars=[f"{value_name}{x}" for x in range(1, 7)],
        var_name=var_name,
        value_name=value_name,
    )


def _normalize_tables(
    clients: pd.DataFrame, return_default: bool = False
) -> List[pd.DataFrame]:
    """Normalize client into several tables.

    Args:
        clients: Source data.
        return_default: Return next month default payment status as a seperate DataFrame.
    Returns:
        Normalized tables as a list of DataFrames

    """
    # Extract monthly repayment status from clients as the base
    profiles = _melt_client_profile(clients, var_name="MONTH", value_name="PAY_")
    profiles["MONTH"] = profiles["MONTH"].str.get(-1)

    # Extract bill amounts from clients
    bill_amounts = _melt_client_profile(
        clients, var_name="MONTH", value_name="BILL_AMT"
    )
    bill_amounts["MONTH"] = bill_amounts["MONTH"].str.get(-1)

    # Merge bill amounts to the base
    profiles = profiles.merge(bill_amounts, how="left", on=["ID", "MONTH"])

    # Extract pay amounts from clients
    pay_amounts = _melt_client_profile(clients, var_name="MONTH", value_name="PAY_AMT")
    pay_amounts["MONTH"] = pay_amounts["MONTH"].str.get(-1)

    # Merge pay amounts to the base
    profiles = profiles.merge(pay_amounts, how="left", on=["ID", "MONTH"])

    profiles["MONTH"] = pd.to_datetime(
        profiles["MONTH"].replace(
            {
                "1": "2005-09-01",
                "2": "2005-08-01",
                "3": "2005-07-01",
                "4": "2005-06-01",
                "5": "2005-05-01",
                "6": "2005-04-01",
            }
        )
    )

    profiles.rename(columns={"PAY_": "PAY_STATUS"}, inplace=True)

    # Drop columns related to profile
    normalized_clients = clients[
        ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    ]

    if return_default:
        labels = clients[["ID", "DEFAULT_PAY"]]
        labels["DEFAULT_PAY"] = labels["DEFAULT_PAY"].astype(np.float32)

        return [normalized_clients, profiles, labels]

    return [normalized_clients, profiles]


def preprocess_clients(clients: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for clients.

    Args:
        clients: Source data.
    Returns:
        Preprocessed clients.

    """
    # Rename columns to proper name
    clients.rename(
        columns={"PAY_0": "PAY_1"},
        inplace=True,
    )

    if "default payment next month" in clients.columns:
        clients.rename(
            columns={"default payment next month": "DEFAULT_PAY"},
            inplace=True,
        )

    # List columns by dtype
    category_cols = clients.select_dtypes("category").columns.tolist()
    numeric_cols = clients.select_dtypes("number").columns.tolist()

    # Set ID as column
    clients.reset_index(inplace=True)
    clients["ID"] = clients["ID"].astype(str)

    # Enforce formatting rules
    clients["SEX"] = clients["SEX"].apply(_enforce_gender_format)
    clients["EDUCATION"] = clients["EDUCATION"].apply(_enforce_education_format)
    clients["MARRIAGE"] = clients["MARRIAGE"].apply(_enforce_marital_format)

    # Enforce dtype
    clients[category_cols] = clients[category_cols].astype("category")
    clients[numeric_cols] = clients[numeric_cols].apply(_downcast_numeric)

    # Enforce repayment status
    for x in range(1, 7):
        col = f"PAY_{x}"
        clients[col] = clients[col].apply(_enforce_repayment_status)

    return clients


def normalize_train_tables(clients: pd.DataFrame) -> List[pd.DataFrame]:
    return _normalize_tables(clients=clients, return_default=True)


# def _is_true(x):
#     return x == "t"


# def _parse_percentage(x):
#     if isinstance(x, str):
#         return float(x.replace("%", "")) / 100
#     return float("NaN")


# def _parse_money(x):
#     return float(x.replace("$", "").replace(",", ""))


# def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess the data for companies.

#     Args:
#         companies: Source data.
#     Returns:
#         Preprocessed data.

#     """

#     companies["iata_approved"] = companies["iata_approved"].apply(_is_true)

#     companies["company_rating"] = companies["company_rating"].apply(_parse_percentage)

#     return companies


# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess the data for shuttles.

#     Args:
#         shuttles: Source data.
#     Returns:
#         Preprocessed data.

#     """
#     shuttles["d_check_complete"] = shuttles["d_check_complete"].apply(_is_true)

#     shuttles["moon_clearance_complete"] = shuttles["moon_clearance_complete"].apply(
#         _is_true
#     )

#     shuttles["price"] = shuttles["price"].apply(_parse_money)

#     return shuttles


# def create_master_table(
#     shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
# ) -> pd.DataFrame:
#     """Combines all data to create a master table.

#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Source data for reviews.
#     Returns:
#         Master table.

#     """
#     rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")

#     with_companies = rated_shuttles.merge(
#         companies, left_on="company_id", right_on="id"
#     )

#     master_table = with_companies.drop(["shuttle_id", "company_id"], axis=1)
#     master_table = master_table.dropna()
#     return master_table
