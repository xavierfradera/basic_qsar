import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    # Build a QSAR model in 8 lines of Python
    # Based on code from https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/QSAR_in_8_lines.ipynb
    #
    import pandas as pd
    import datamol as dm
    from molfeat.calc import FPCalculator
    from molfeat.trans import MoleculeTransformer
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingRegressor
    from yellowbrick.regressor import prediction_error, residuals_plot
    import wget

    return (
        FPCalculator,
        HistGradientBoostingRegressor,
        MoleculeTransformer,
        dm,
        np,
        pd,
        prediction_error,
        residuals_plot,
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
def _(activity_col, model, np, prediction_error, test, train):
    # **8.** Use YellowBrick to build a model and visualize its performance.
    # The Loss reported in the plot below is the R^2 for the model.
    visualizer = prediction_error(model, np.stack(train.fp), train[activity_col], np.stack(test.fp), test[activity_col])
    return


@app.cell
def _(activity_col, model, np, residuals_plot, test, train):
    # Bonus
    # Plot the residuals for the training and test sets
    viz = residuals_plot(model, np.stack(train.fp), train[activity_col], np.stack(test.fp), test[activity_col], is_fitted=True)
    return


if __name__ == "__main__":
    app.run()
