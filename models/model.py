from typing import List
from typing import Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """
    df['day'] = pd.to_datetime(df['day'])
    df_test = df[df['day'] >= max(df['day']) - timedelta(days=test_days)]
    df_train = df.loc[df.index.difference(df_test.index)]
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

class MultiTargetModel:
    def __init__(
        self,
        features: List[str],
        horizons: List[int] = [7, 14, 21],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
                sku_id_2: {
                    (quantile_1, horizon_1): model_3,
                    (quantile_1, horizon_2): model_4,
                    ...
                },
                ...
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]

        self.fitted_models_ = {}

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for fitting and predicting.

        Parameters
        ----------
        data : pd.DataFrame
            Data to prepare.

        Returns
        -------
        pd.DataFrame
            Prepared data.
        """
        df = data.copy()
        df.dropna(inplace=True)
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        df.set_index([self.sku_col, self.date_col], inplace=True)
        df.sort_index(inplace=True)

        desired_columns = self.features + self.targets
        df = df.filter(items=desired_columns)

        return df
    
    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        df = self._prepare_data(data)

        for sku_id in tqdm(df.index.get_level_values(0).unique(), disable=not verbose):
            df_sku = df.loc[pd.IndexSlice[sku_id, :], :]  # type: ignore
            df_features = df_sku[self.features]

            models_one_sku = {}
            for horizon in self.horizons:
                for quantile in self.quantiles:
                    target_name = f"next_{horizon}d"
                    y = df_sku[target_name]

                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=0,
                        solver="highs",
                    )
                    model.fit(df_features, y)

                    models_one_sku[(quantile, horizon)] = model

            self.fitted_models_[sku_id] = models_one_sku

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        df = self._prepare_data(data)

        predictions = []
        for sku_id in df.index.get_level_values(0).unique():
            sku_df = df.loc[pd.IndexSlice[sku_id, :], :]  # type: ignore

            if self.targets[0] in sku_df.columns:
                X_test = sku_df.drop(columns=self.targets)
            else:
                X_test = sku_df

            rows_count = X_test.shape[0]

            sku_models = self.fitted_models_.get(sku_id, None)

            predictions_one_sku = []
            for horizon in self.horizons:
                for quantile in self.quantiles:
                    if sku_models is not None:
                        model = sku_models[(quantile, horizon)]
                        y_pred = model.predict(X_test)
                    else:
                        y_pred = np.zeros(rows_count)
                    predictions_one_sku.append(y_pred)
            predictions.append(np.stack(predictions_one_sku, axis=1))

        predictions = np.concatenate(predictions, axis=0)
        predictions = pd.DataFrame(
            predictions,
            index=df.index,
            columns=[
                f"pred_{horizon}d_q{int(quantile*100)}"
                for quantile in self.quantiles
                for horizon in self.horizons
            ],
        )

        predictions.reset_index(inplace=True)
        return predictions