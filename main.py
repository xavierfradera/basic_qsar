import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    # Basic QSAR models
    # All imports made here are returned so every import is global and
    # reusable from any other cell in the notebook, instead of each cell
    # re-importing what it needs.
    import marimo as mo
    import pandas as pd
    import datamol as dm
    from molfeat.calc import FPCalculator
    from molfeat.trans import MoleculeTransformer
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    import wget

    return (
        FPCalculator,
        MoleculeTransformer,
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
def _(pd):
    # Read the data into a Pandas dataframe.

    filename = "carbonic.csv"

    df = pd.read_csv(filename)
    activity_col = df.columns[-1]
    print(f"The activity column is proably {activity_col}")
    df.head()
    return activity_col, df


@app.cell
def _():
    # List of FPs and regressors we want to try
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    fp_list = [  'ecfp', 'fcfp' ] #,  'topological', 'atompair']
    reg_list = [HistGradientBoostingRegressor, RandomForestRegressor] #, LGBMRegressor, XGBRegressor]
    print("%i fingerprint types and %i regressor types: % i combinations" % (len(fp_list), len(reg_list), len(fp_list)*len(reg_list)))
    return fp_list, reg_list


@app.cell
def _(mean_absolute_error, r2_score, root_mean_squared_error, stats):
    # **Quality of prediction.** Regress predicted activity against actual
    # activity for the train and test sets and report R^2, Pearson r, MAE,
    # RMSE, and the regression line's slope/intercept.
    def regression_stats(actual, predicted):
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted)
        return {
            "r_squared": r2_score(actual, predicted),
            "pearson_r": r_value,
            "mae": mean_absolute_error(actual, predicted),
            "rmse": root_mean_squared_error(actual, predicted),
            "slope": slope,
            "intercept": intercept,
        }

    return (regression_stats,)


@app.cell
def _(df, train_test_split):
    # training and test sets
    train, test = train_test_split(df)
    print("Train set: %i\nTest set: %i" % (len(train), len(test)))
    return test, train


@app.cell
def _(
    FPCalculator,
    MoleculeTransformer,
    activity_col,
    dm,
    fp_list,
    np,
    pd,
    reg_list,
    regression_stats,
    test,
    train,
):
    # Loop over all combinations and create all models
    # Calculate stats for each model and save in dataframe
    models = {}
    # Actual/predicted values per model, kept for the scatter plot cell so
    # it doesn't need to refit or re-transform anything.
    predictions = {}
    _cols = { 'model_number': 'int16',
              'split': 'string',
              'r_squared': 'float64',
              'pearson_r': 'float64',
              'mae': 'float64',
              'rmse': 'float64',
              'slope': 'float64',
              'intercept': 'float64',
            }
    summary = pd.DataFrame(columns=_cols.keys()).astype(_cols)
    _train_act = train[activity_col]
    _test_act = test[activity_col]

    _i = 0
    _train2 = train.copy()
    _test2 = test.copy()
    for _f in fp_list:
        _calc = FPCalculator(_f)
        _trans = MoleculeTransformer(_calc)
        with dm.without_rdkit_log():
            _train2['fp'] = _trans.transform(_train2.SMILES.values)
            _test2['fp'] = _trans.transform(_test2.SMILES.values)
        for _r in reg_list:
            print("%i: %s - %s" % (_i, _f, _r))
            _model = _r()
            _model.fit(np.stack(_train2.fp), _train2[activity_col])

            _i += 1
            # stats - should put all into a dataframe
            _train_pred = _model.predict(np.stack(_train2.fp))
            _test_pred = _model.predict(np.stack(_test2.fp))
            _s = regression_stats(_train_act, _train_pred)
            _s['model_number'] = _i
            _s['split'] = 'train'
            _add = pd.DataFrame([_s])
            summary = pd.concat([summary, _add], ignore_index=True)
            _s = regression_stats(_test_act, _test_pred)
            _s['model_number'] = _i
            _s['split'] = 'test'
            _add = pd.DataFrame([_s])
            summary = pd.concat([summary, _add], ignore_index=True)
            # save model and its predictions
            models[_i] = _model
            predictions[_i] = {
                'fp': _f,
                'regressor': _r.__name__,
                'train_actual': _train_act.to_numpy(),
                'train_pred': _train_pred,
                'test_actual': _test_act.to_numpy(),
                'test_pred': _test_pred,
            }
    n_models = _i
    return predictions, summary


@app.cell
def _(mo, summary):
    # display summary
    _numeric_cols = ["r_squared", "pearson_r", "mae", "rmse", "slope", "intercept"]

    _pivoted = summary.pivot(index="model_number", columns="split", values=_numeric_cols)
    _pivoted.columns = [f"{_metric}_{_split}" for _metric, _split in _pivoted.columns]
    _pivoted = _pivoted.reset_index()

    _value_cols = [_c for _c in _pivoted.columns if _c != "model_number"]

    mo.ui.table(
        _pivoted,
        format_mapping={_col: "{:.2f}" for _col in _value_cols},
        selection=None,
    )
    return


@app.cell
def _(mo, predictions, summary):
    # Model selector, defaulting to the model with the highest test R^2
    _test_summary = summary[summary["split"] == "test"]
    _default_model = int(_test_summary.loc[_test_summary["r_squared"].idxmax(), "model_number"])

    _options = {
        f"{_n}: {_p['fp']} + {_p['regressor']}": _n for _n, _p in predictions.items()
    }
    _default_label = next(_label for _label, _n in _options.items() if _n == _default_model)

    model_selector = mo.ui.dropdown(options=_options, value=_default_label, label="Model")
    model_selector
    return (model_selector,)


@app.cell
def _(model_selector, np, plt, predictions, summary):
    # scatter plots - actual vs. predicted for the selected model's
    # train and test sets
    _model_number = model_selector.value
    _pred = predictions[_model_number]
    _model_label = f"{_pred['fp']} + {_pred['regressor']}"

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    for _ax, _split in zip(_axes, ["train", "test"]):
        _actual = _pred[f"{_split}_actual"]
        _predicted = _pred[f"{_split}_pred"]
        _stats = summary[
            (summary["model_number"] == _model_number) & (summary["split"] == _split)
        ].iloc[0]

        _lims = [min(_actual.min(), _predicted.min()), max(_actual.max(), _predicted.max())]
        _xs = np.linspace(_lims[0], _lims[1], 2)

        _ax.scatter(_actual, _predicted, alpha=0.6, edgecolor="k", linewidth=0.3, label="Predictions")
        _ax.plot(_xs, _xs, "k--", label="Equality (y = x)")
        _ax.plot(_xs, _xs + 1, color="gray", linestyle=":", label="±1 unit")
        _ax.plot(_xs, _xs - 1, color="gray", linestyle=":")
        _ax.plot(_xs, _stats["slope"] * _xs + _stats["intercept"], "r-", label="Correlation line")

        _ax.set_xlim(_lims)
        _ax.set_ylim(_lims)
        _ax.set_xlabel("Actual")
        _ax.set_ylabel("Predicted")
        _ax.set_title(f"{_split.capitalize()} (R² = {_stats['r_squared']:.2f})")
        _ax.legend()

    _fig.suptitle(f"Model {_model_number}: {_model_label}")
    _fig
    return


if __name__ == "__main__":
    app.run()
