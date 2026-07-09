import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    # Build a QSAR model in 8 lines of Python
    # Based on code from https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/QSAR_in_8_lines.ipynb
    #
    import marimo as mo
    import pandas as pd
    import datamol as dm
    from molfeat.calc import FPCalculator
    from molfeat.trans import MoleculeTransformer
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import (
        PredictionErrorDisplay,
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )
    import wget

    return (
        FPCalculator,
        HistGradientBoostingRegressor,
        MoleculeTransformer,
        PredictionErrorDisplay,
        dm,
        mean_absolute_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        stats,
        train_test_split,
    )


@app.cell
def _():
    # **0.** Get an input file.
    # We  need a csv file with Chemical structures as SMILES and some
    # property or actvity file.  The file should have a column named SMILES and an
    # activity column whose name we will specify below.
    #
    # The code below will download a demo file called "carbonic.csv" from GitHub.

    # wget.download("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/carbonic.csv")
    return


@app.cell
def _(pd):
    # **1.** Read the data into a Pandas dataframe.
    # In the next cell we read the data from a csv file and put it into a Pandas
    # dataframe
    # The code below uses the last column in the dataframe as the the acivity column.

    filename = "carbonic.csv"
    df = pd.read_csv(filename)
    activity_col = df.columns[-1]
    print(f"The activity column is proably {activity_col}")
    df.head()
    return activity_col, df


@app.cell
def _(FPCalculator):
    # **2.** Instantiate a Fingerprint calculator from the awesome molfeat
    # package. This package has several descriptor types available.
    #     from molfeat.calc import FP_FUNCS
    #     FP_FUNCS.keys()
    #     dict_keys(['maccs', 'avalon', 'ecfp', 'fcfp', 'topological', 'atompair',
    #     'rdkit', 'pattern', 'layered', 'map4', 'secfp', 'erg', 'estate',
    #     'avalon-count', 'rdkit-count', 'ecfp-count', 'fcfp-count',
    #     'topological-count', 'atompair-count'])
    calc = FPCalculator("ecfp")
    return (calc,)


@app.cell
def _(MoleculeTransformer, calc):
    # **3.** Instantiate a molecule transfomer from molfeat.
    # This object takes a list of SMILES as input and returns descriptors.  It's
    # very flexible and can run in parallel.
    trans = MoleculeTransformer(calc)
    return (trans,)


@app.cell
def _(df, dm, trans):
    # **4-5.** Calculate the fingerprints.
    # Note the use of the function from datamol that silences logging messages
    # from the RDKit.  This is a more polite version of my rd_shut_the_hell_up
    # function in useful_rdkit_utils.
    with dm.without_rdkit_log():
        df['fp'] = trans.transform(df.SMILES.values)
    return


@app.cell
def _(df, train_test_split):
    # **6.** Split the data into training and test sets.
    # I like to do this with dataframes.  That way I don't have to remember the
    # order in which train_X, test_X, train_y, and test_y are returned by
    # train_test_split.
    train, test = train_test_split(df)
    return test, train


@app.cell
def _(HistGradientBoostingRegressor):
    # **7.** Instantiate an sklearn style regressor.
    # In this case I used HistGradientBoostingRegressor, which is the
    # scikit-learn implementation of LightGBM.  You can easily plug in any
    # scikit-learn compatible regressor like RandomForest or XGBoost.
    #     from lightgbm import LGBMRegressor
    #     model = LGBMRegressor()
    #     from sklearn.ensemble import RandomForestRegressor
    #     model = RandomForestRegressor()
    #     from xgboost import XGBRegressor
    #     model = XGBRegressor()
    model = HistGradientBoostingRegressor()
    return (model,)


@app.cell
def _(PredictionErrorDisplay, activity_col, model, np, plt, test, train):
    # **8.** Fit the model and visualize its performance with scikit-learn's
    # PredictionErrorDisplay (actual vs. predicted) for the train and test sets.
    model.fit(np.stack(train.fp), train[activity_col])
    fitted_model = model

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    PredictionErrorDisplay.from_estimator(
        fitted_model, np.stack(train.fp), train[activity_col],
        kind="actual_vs_predicted", ax=axes[0],
    )
    axes[0].set_title("Train")
    PredictionErrorDisplay.from_estimator(
        fitted_model, np.stack(test.fp), test[activity_col],
        kind="actual_vs_predicted", ax=axes[1],
    )
    axes[1].set_title("Test")
    fig.tight_layout()
    return (fitted_model,)


@app.cell
def _(
    PredictionErrorDisplay,
    activity_col,
    fitted_model,
    np,
    plt,
    test,
    train,
):
    # Bonus
    # Plot the residuals for the training and test sets
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    PredictionErrorDisplay.from_estimator(
        fitted_model, np.stack(train.fp), train[activity_col],
        kind="residual_vs_predicted", ax=axes2[0],
    )
    axes2[0].set_title("Train")
    PredictionErrorDisplay.from_estimator(
        fitted_model, np.stack(test.fp), test[activity_col],
        kind="residual_vs_predicted", ax=axes2[1],
    )
    axes2[1].set_title("Test")
    fig2.tight_layout()
    return


@app.cell(hide_code=True)
def _(activity_col, df, fitted_model, mo, np, test, train):
    # Summary. Only runs once `fitted_model` exists, i.e. step 8 has
    # successfully fit the model.
    predictions = df.copy()
    predictions["split"] = np.where(predictions.index.isin(train.index), "train", "test")
    predictions["predicted"] = fitted_model.predict(np.stack(predictions.fp))
    predictions["residual"] = predictions[activity_col] - predictions["predicted"]

    summary_md = mo.md(
        f"""
        ## Summary

        - **Total molecules:** {len(df)}
        - **Train set size:** {len(train)}
        - **Test set size:** {len(test)}
        """
    )

    results_table = mo.ui.table(
        predictions[["SMILES", activity_col, "predicted", "residual", "split"]],
        selection=None,
    )

    mo.vstack([summary_md, results_table])
    return (predictions,)


@app.cell
def _(
    activity_col,
    mean_absolute_error,
    mo,
    pd,
    predictions,
    r2_score,
    root_mean_squared_error,
    stats,
):
    # **Quality of prediction.** Regress predicted activity against actual
    # activity for the train and test sets and report R^2, Pearson r, MAE,
    # RMSE, and the regression line's slope/intercept.
    def _regression_stats(actual, predicted):
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted)
        return {
            "r_squared": r2_score(actual, predicted),
            "pearson_r": r_value,
            "mae": mean_absolute_error(actual, predicted),
            "rmse": root_mean_squared_error(actual, predicted),
            "slope": slope,
            "intercept": intercept,
        }

    fit_stats = {
        split_name: _regression_stats(
            subset[activity_col].to_numpy(), subset["predicted"].to_numpy()
        )
        for split_name, subset in predictions.groupby("split")
    }

    fit_stats_df = pd.DataFrame(fit_stats).T.reset_index(names="split")
    mo.ui.table(fit_stats_df, selection=None)
    return (fit_stats,)


@app.cell
def _(activity_col, fit_stats, np, plt, predictions):
    # Scatter plot of actual vs. predicted activity for train and test, with
    # the fitted regression line and the y = x equality line for reference.
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))
    for ax, split_name in zip(axes3, ["train", "test"]):
        subset = predictions[predictions["split"] == split_name]
        actual = subset[activity_col].to_numpy()
        predicted = subset["predicted"].to_numpy()

        lims = [
            min(actual.min(), predicted.min()),
            max(actual.max(), predicted.max()),
        ]
        xs = np.linspace(lims[0], lims[1], 2)

        ax.scatter(actual, predicted, alpha=0.6, edgecolor="k", linewidth=0.3)
        ax.plot(xs, xs, "k--", label="y = x")
        ax.plot(
            xs,
            fit_stats[split_name]["slope"] * xs + fit_stats[split_name]["intercept"],
            "r-",
            label="Regression fit",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Actual {activity_col}")
        ax.set_ylabel(f"Predicted {activity_col}")
        ax.set_title(f"{split_name.capitalize()} (R² = {fit_stats[split_name]['r_squared']:.3f})")
        ax.legend()
    fig3.tight_layout()
    return


if __name__ == "__main__":
    app.run()
