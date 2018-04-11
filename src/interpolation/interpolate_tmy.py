#interpolation on TMY data using GP regression
import pvlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RationalQuadratic
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interpolate_column(data, name, n_samples=500,):
    min_since_epoch = data['GHI'].index[0:n_samples].view('int64') // pd.Timedelta(1, unit='m')

    col_data = data[name].as_matrix()[0:n_samples]

    #create GP regressor
    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) +  Matern(length_scale=2, nu=3/2) + RationalQuadratic()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gpr.fit(min_since_epoch.reshape(-1, 1), col_data)

    #create minute by minute index
    date_range = pd.date_range(start=data[name].index[0], end=data[name].index[n_samples] , freq="T")
    date_range_min = date_range.view('int64') // pd.Timedelta(1, unit='m')

    interpolated_data = gpr.predict(date_range_min.reshape(-1, 1))

    #remove all values less than zero

    interpolated_data[interpolated_data < 0] = 0

    inter_series = pd.Series(interpolated_data, index=date_range, name="{}".format(name))

    return inter_series


def run():
    loc = "/Users/edwardwilliams/Documents/research/heliotrope/simulations/data/722745TYA.CSV"
    tmy_data, meta = pvlib.tmy.readtmy3(filename=loc)

    #TODO: add cloud cover
    cols_to_interpolate = ['DHI', 'DNI', 'GHI','Wspd', 'DryBulb', 'TotCld', 'OpqCld']
    n_samples = 500

    true = [tmy_data[col][tmy_data[col].index[0]:tmy_data[col].index[n_samples]] for col in cols_to_interpolate]
    interpolated = [interpolate_column(tmy_data, col) for col in cols_to_interpolate]

    df = pd.concat(interpolated, axis=1)
    plt.figure(figsize=(50,50))
    df.plot();
    plt.savefig("interpolated_all.png")

    df_true = pd.concat(true, axis=1)
    plt.figure(figsize=(10,10))
    df_true.plot();
    plt.savefig("true_all.png")




if __name__=="__main__":
    run()
