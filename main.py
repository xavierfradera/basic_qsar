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

    from sklearn.metrics import (
        PredictionErrorDisplay,
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )
    import wget

    return (
        mean_absolute_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        stats,
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
    return (activity_col,)


@app.cell
def _():
    # Create list of FPs and regressors we want to try
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    fp_list = [  'ecfp', 'fcfp',  'topological', 'atompair']
    reg_list = [HistGradientBoostingRegressor, RandomForestRegressor, LGBMRegressor, XGBRegressor]
    print("%i fingerprint types and %i regressor types: % i combinations" % (len(fp_list), len(reg_list), len(fp_list)*len(reg_list)))
    return (reg_list,)


app._unparsable_cell(
    r"""
    # Loop over all combinations
    models = []
    for _f in fp_list:
        for _r in reg_list:
            print(_f, _r)
            _calc = FPCalculator(_f)
            _trans = MoleculeTransformer(_calc)
            with dm.without_rdkit_log():
                df['fp'] = _trans.transform(df.SMILES.values)
            _df_fp = df
            _train, _test = train_test_split(_df_fp)
            _r.fit(np.stack(_train.fp), _train[activity_col])
            fitted_model = model

            ## calculate regression etc. and put in array
            models.append([ _f, _r, .......])
    """,
    name="_"
)


@app.cell
def _(reg_list):
    r =reg_list[0]
    print(r)
    return


@app.cell
def _(activity_col, df_fp, fitted_model, mo, np, test, train):
    # Summary. Only runs once `fitted_model` exists, i.e. step 8 has
    # successfully fit the model.
    predictions = df_fp.copy()
    predictions["split"] = np.where(predictions.index.isin(train.index), "train", "test")
    predictions["predicted"] = fitted_model.predict(np.stack(predictions.fp))
    predictions["residual"] = predictions[activity_col] - predictions["predicted"]

    summary_md = mo.md(
        f"""
        ## Summary

        - **Total molecules:** {len(df_fp)}
        - **Train set size:** {len(train)}
        - **Test set size:** {len(test)}
        """
    )

    results_table = mo.ui.table(
        predictions[["SMILES", activity_col, "predicted", "residual", "split"]],
        selection=None,
        format_mapping={ activity_col: lambda v: round(v,2),
                         "predicted": lambda v: round(v,2),
                         "residual": lambda v: round(v,2)                     
                       }
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
    mo.ui.table(fit_stats_df, selection=None,
               format_mapping={ 'r_squared': lambda v: round(v,2),
                                'pearson_r': lambda v: round(v,2),
                                'mae': lambda v: round(v,2),
                                'rmse': lambda v: round(v,2),
                                'slope': lambda v: round(v,2),
                                'intercept': lambda v: round(v,2)
                              })
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
        ax.plot(xs, xs, "k", label="y = x")
        ax.plot(xs, xs+1, "k--", label="+1")
        ax.plot(xs, xs-1, "k--", label="-1")
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
    fig3
    return


if __name__ == "__main__":
    app.run()
