#-*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
#import xgboost


register_matplotlib_converters()
# use `fc-list` at cmd to see the available fonts at your computer
font_prop = mfm.FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
#font_prop = mfm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")


# lagged cross correlation between 1D signals x(t) and y(t-l)
def lagged_cross_correlation(x, y, lag):
    mu_x = np.mean(x[lag:])
    mu_y = np.mean(y[:-lag])
    var_x = np.var(x[lag:])
    var_y = np.var(y[:-lag])
    lcc = np.mean((x[lag:]-mu_x)*(y[:-lag]-mu_y))/np.sqrt(var_x*var_y)
    return lcc


def svr_forecast(c_delta, r_delta, forecast_length, with_province):
    num_obs = c_delta.shape[0]
    num_state = c_delta.shape[1]
    param_grid = {"C": [0.1, 1, 10, 100, 1000, 10000, 100000], "epsilon": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    if with_province:
        # New confirmed estimation
        csvr_list = [GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False) for i in range(0, num_state)]
        # --Training
        for i in range(0, num_state):
            csvr_list[i].fit(X=c_delta[:-1, :], y=c_delta[1:, i])
        c_fit = []
        for d in range(0, num_obs-1):
            c_fit.append([csvr_list[i].predict(X=c_delta[d, :].reshape(1, -1))[0] for i in range(0, num_state)])
        c_fit = np.sum(c_fit, axis=1).astype(int)
        print("The MAE of the corr-svr fitting for new-confirmed is: ", mean_absolute_error(np.sum(c_delta[1:, :], axis=1), c_fit))
        # --Forecasting
        c_forecast = [[csvr_list[i].predict(X=c_delta[-1, :].reshape(1, -1))[0] for i in range(0, num_state)]]
        for d in range(1, forecast_length):
            c_forecast.append([csvr_list[i].predict(X=np.array(c_forecast[d-1]).reshape(1, -1))[0] for i in range(0, num_state)])
        c_forecast = np.sum(c_forecast, axis=1).astype(int)
        # New recovery estimation
        rsvr_list = [GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False) for i in range(0, num_state)]
        # --Training
        for i in range(0, num_state):
            rsvr_list[i].fit(X=r_delta[:-1, :], y=r_delta[1:, i])
        r_fit = []
        for d in range(0, num_obs - 1):
            r_fit.append([rsvr_list[i].predict(X=r_delta[d, :].reshape(1, -1))[0] for i in range(0, num_state)])
        r_fit = np.sum(r_fit, axis=1).astype(int)
        print("The MAE of the corr-svr fitting for new-recovery is: ", mean_absolute_error(np.sum(r_delta[1:, :], axis=1), r_fit))
        # --Forecasting
        r_forecast = [[rsvr_list[i].predict(X=r_delta[-1, :].reshape(1, -1))[0] for i in range(0, num_state)]]
        for d in range(1, forecast_length):
            r_forecast.append([rsvr_list[i].predict(X=np.array(r_forecast[d - 1]).reshape(1, -1))[0] for i in range(0, num_state)])
        r_forecast = np.sum(r_forecast, axis=1).astype(int)
    else:
        # New confirmed estimation
        csvr = GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False)
        c_delta = np.sum(c_delta, axis=1)
        # --Training
        csvr.fit(X=c_delta[:-1].reshape(-1, 1), y=c_delta[1:])
        c_fit = csvr.predict(c_delta[:-1].reshape(-1, 1))
        print("The MAE of the svr fitting for new-confirmed is: ", mean_absolute_error(c_delta[1:], c_fit))
        # --Forecasting
        c_forecast = [csvr.predict(X=c_delta[-1].reshape(1, -1))]
        for d in range(1, forecast_length):
            c_forecast.append(csvr.predict(X=np.array(c_forecast[d-1]).reshape(1, -1))[0])
        c_forecast = np.array(c_forecast).astype(int)
        # New recovery estimation
        rsvr = GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False)
        r_delta = np.sum(r_delta, axis=1)
        # --Training
        rsvr.fit(X=r_delta[:-1].reshape(-1, 1), y=r_delta[1:])
        r_fit = rsvr.predict(r_delta[:-1].reshape(-1, 1))
        print("The MAE of the svr fitting for new-recovery is: ", mean_absolute_error(r_delta[1:], r_fit))
        # --Forecasting
        r_forecast = [rsvr.predict(X=r_delta[-1].reshape(1, -1))]
        for d in range(1, forecast_length):
            r_forecast.append(rsvr.predict(X=np.array(r_forecast[d - 1]).reshape(1, -1))[0])
        r_forecast = np.array(r_forecast).astype(int)
    return c_fit, c_forecast, r_fit, r_forecast


def corrsvr_forecast(c_delta, r_delta, c_ccmat, r_ccmat, forecast_length, with_province, kmax):
    num_obs = c_delta.shape[0]
    num_state = c_delta.shape[1]
    param_grid = {"C": [0.1, 1, 10, 100, 1000, 10000, 100000], "epsilon": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    if with_province:
        # New confirmed estimation
        c_select = np.argpartition(-np.abs(c_ccmat), axis=1, kth=kmax)[:, 0:kmax]
        csvr_list = [GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False) for i in range(0, num_state)]
        # --Training
        for i in range(0, num_state):
            csvr_list[i].fit(X=c_delta[:-1, c_select[i]], y=c_delta[1:, i])
        c_fit = []
        for d in range(0, num_obs-1):
            c_fit.append([csvr_list[i].predict(X=c_delta[d, c_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
        c_fit = np.sum(c_fit, axis=1).astype(int)
        print("The MAE of the corr-svr fitting for new-confirmed is: ", mean_absolute_error(np.sum(c_delta[1:, :], axis=1), c_fit))
        # --Forecasting
        c_forecast = [[csvr_list[i].predict(X=c_delta[-1, c_select[i]].reshape(1, -1))[0] for i in range(0, num_state)]]
        for d in range(1, forecast_length):
            c_forecast.append([csvr_list[i].predict(X=np.array(c_forecast[d-1])[c_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
        c_forecast = np.sum(c_forecast, axis=1).astype(int)
        # New recovery estimation
        r_select = np.argpartition(-np.abs(r_ccmat), axis=1, kth=kmax)[:, 0:kmax]
        rsvr_list = [GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False) for i in range(0, num_state)]
        # --Training
        for i in range(0, num_state):
            rsvr_list[i].fit(X=r_delta[:-1, r_select[i]], y=r_delta[1:, i])
        r_fit = []
        for d in range(0, num_obs - 1):
            r_fit.append([rsvr_list[i].predict(X=r_delta[d, r_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
        r_fit = np.sum(r_fit, axis=1).astype(int)
        print("The MAE of the corr-svr fitting for new-recovery is: ", mean_absolute_error(np.sum(r_delta[1:, :], axis=1), r_fit))
        # Forecasting
        r_forecast = [[rsvr_list[i].predict(X=r_delta[-1, r_select[i]].reshape(1, -1))[0] for i in range(0, num_state)]]
        for d in range(1, forecast_length):
            r_forecast.append([rsvr_list[i].predict(X=np.array(r_forecast[d - 1])[r_select[i]].reshape(1, -1))[0] for i in
                 range(0, num_state)])
        r_forecast = np.sum(r_forecast, axis=1).astype(int)
    else:
        # New confirmed estimation
        csvr = GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False)
        c_delta = np.sum(c_delta, axis=1)
        # --Training
        csvr.fit(X=c_delta[:-1].reshape(-1, 1), y=c_delta[1:])
        c_fit = csvr.predict(c_delta[:-1].reshape(-1, 1))
        print("The MAE of the corr-svr fitting for new-confirmed is: ", mean_absolute_error(c_delta[1:], c_fit))
        # --Forecasting
        c_forecast = [csvr.predict(X=c_delta[-1].reshape(1, -1))]
        for d in range(1, forecast_length):
            c_forecast.append(csvr.predict(X=np.array(c_forecast[d-1]).reshape(1, -1))[0])
        c_forecast = np.array(c_forecast).astype(int)
        # New recovery estimation
        rsvr = GridSearchCV(SVR(kernel="rbf", gamma="scale"), param_grid, cv=5, iid=False)
        r_delta = np.sum(r_delta, axis=1)
        # --Training
        rsvr.fit(X=r_delta[:-1].reshape(-1, 1), y=r_delta[1:])
        r_fit = rsvr.predict(r_delta[:-1].reshape(-1, 1))
        print("The MAE of the corr-svr fitting for new-recovery is: ", mean_absolute_error(r_delta[1:], r_fit))
        # --Forecasting
        r_forecast = [rsvr.predict(X=r_delta[-1].reshape(1, -1))]
        for d in range(1, forecast_length):
            r_forecast.append(rsvr.predict(X=np.array(r_forecast[d - 1]).reshape(1, -1))[0])
        r_forecast = np.array(r_forecast).astype(int)
    return c_fit, c_forecast, r_fit, r_forecast



def xgboost_forecast(c_delta, r_delta, forecast_length):
    num_obs = c_delta.shape[0]
    num_state = c_delta.shape[1]
    param_grid = {"n_estimators": [3, 5, 7, 9], "max_depth": [1, 2, 3, 4, 5], "learning_rate": [0.001, 0.01, 0.1, 1],
                  "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9]}
    # New confirmed estimation
    cxgb_list = [GridSearchCV(xgboost.XGBRegressor(objective='reg:squarederror'), param_grid) for i in range(0, num_state)]
    # --Training
    for i in range(0, num_state):
        # data: num_data * num_feature
        cxgb_list[i].fit(X=c_delta[:-1, :], y=c_delta[1:, i])
    c_fit = []
    for d in range(0, num_obs - 1):
        c_fit.append([cxgb_list[i].predict(c_delta[d, :].reshape(1, -1))[0] for i in range(0, num_state)])
    c_fit = np.sum(c_fit, axis=1).astype(int)
    print("The MAE of the xgboost fitting for new-confirmed is: ", mean_absolute_error(np.sum(c_delta[1:, :], axis=1), c_fit))
    # --Forecasting
    c_forecast = [[cxgb_list[i].predict(X=np.array(c_delta[-1, :]).reshape(1, -1))[0] for i in range(0, num_state)]]
    for d in range(1, forecast_length):
        c_forecast.append([cxgb_list[i].predict(np.array(c_forecast[d - 1]).reshape(1, -1))[0] for i in
                           range(0, num_state)])
    c_forecast = np.sum(c_forecast, axis=1).astype(int)
    # New recovery estimation
    rxgb_list = [GridSearchCV(xgboost.XGBRegressor(objective='reg:squarederror'), param_grid) for i in range(0, num_state)]
    # --Training
    for i in range(0, num_state):
        # data: num_data * num_feature
        rxgb_list[i].fit(X=r_delta[:-1, :], y=r_delta[1:, i])
    r_fit = []
    for d in range(0, num_obs - 1):
        r_fit.append([rxgb_list[i].predict(r_delta[d, :].reshape(1, -1))[0] for i in range(0, num_state)])
    r_fit = np.sum(r_fit, axis=1).astype(int)
    print("The MAE of the corr-svr fitting for new-recovery is: ", mean_absolute_error(np.sum(r_delta[1:, :], axis=1), r_fit))
    # --Forecasting
    r_forecast = [[rxgb_list[i].predict(X=np.array(r_delta[-1, :]).reshape(1, -1))[0] for i in range(0, num_state)]]
    for d in range(1, forecast_length):
        r_forecast.append([rxgb_list[i].predict(np.array(r_forecast[d - 1]).reshape(1, -1))[0] for i in range(0, num_state)])
    r_forecast = np.sum(r_forecast, axis=1).astype(int)
    return c_fit, c_forecast, r_fit, r_forecast


def corrxgboost_forecast(c_delta, r_delta, c_ccmat, r_ccmat, forecast_length, kmax):
    c_select = np.argpartition(-np.abs(c_ccmat), axis=1, kth=kmax)[:, 0:kmax]
    num_obs = c_delta.shape[0]
    num_state = c_delta.shape[1]
    param_grid = {"n_estimators": [3, 5, 7, 9], "max_depth": [1, 2, 3, 4, 5], "learning_rate": [0.001, 0.01, 0.1, 1],
                  "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9]}
    # New confirmed estimation
    cxgb_list = [GridSearchCV(xgboost.XGBRegressor(objective='reg:squarederror'), param_grid) for i in range(0, num_state)]
    # --Training
    for i in range(0, num_state):
        # data: num_data * num_feature
        cxgb_list[i].fit(X=c_delta[:-1, c_select[i]], y=c_delta[1:, i])
    c_fit = []
    for d in range(0, num_obs-1):
        c_fit.append([cxgb_list[i].predict(c_delta[d, c_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
    c_fit = np.sum(c_fit, axis=1).astype(int)
    print("The MAE of the corr-xgboost fitting for new-confirmed is: ", mean_absolute_error(np.sum(c_delta[1:, :], axis=1), c_fit))
    # --Forecasting
    c_forecast = [[cxgb_list[i].predict(X=np.array(c_delta[-1, c_select[i]]).reshape(1, -1))[0] for i in range(0, num_state)]]
    for d in range(1, forecast_length):
        c_forecast.append([cxgb_list[i].predict(np.array(c_forecast[d-1])[c_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
    c_forecast = np.sum(c_forecast, axis=1).astype(int)
    # New recovery estimation
    r_select = np.argpartition(-np.abs(r_ccmat), axis=1, kth=kmax)[:, 0:kmax]
    rxgb_list = [GridSearchCV(xgboost.XGBRegressor(objective='reg:squarederror'), param_grid) for i in range(0, num_state)]
    # --Training
    for i in range(0, num_state):
        # data: num_data * num_feature
        rxgb_list[i].fit(X=r_delta[:-1, r_select[i]], y=r_delta[1:, i])
    r_fit = []
    for d in range(0, num_obs - 1):
        r_fit.append([rxgb_list[i].predict(r_delta[d, r_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
    r_fit = np.sum(r_fit, axis=1).astype(int)
    print("The MAE of the corr-xgboost fitting for new-recovery is: ", mean_absolute_error(np.sum(r_delta[1:, :], axis=1), r_fit))
    # --Forecasting
    r_forecast = [[rxgb_list[i].predict(X=np.array(r_delta[-1, r_select[i]]).reshape(1, -1))[0] for i in range(0, num_state)]]
    for d in range(1, forecast_length):
        r_forecast.append([rxgb_list[i].predict(np.array(r_forecast[d - 1])[r_select[i]].reshape(1, -1))[0] for i in range(0, num_state)])
    r_forecast = np.sum(r_forecast, axis=1).astype(int)
    return c_fit, c_forecast, r_fit, r_forecast


df = pd.read_csv("data_with_province_update_20200223.csv")

province = df.columns.values[2:]
num_province = len(province)
province_dict = {"hubei": "鄂", "guangdong": "粤", "henan": "豫", "zhejiang": "浙", "hunan": "湘",
                "anhui": "皖", "jiangxi": "赣", "jiangsu": "苏", "chongqing": "渝", "shandong": "鲁",
                "sichuan": "川", "heilongjiang": "黑", "beijing": "京", "shanghai": "沪", "fujian": "闽",
                "hebei": "冀", "shaanxi": "陕", "guangxi": "桂", "yunnan": "云", "hainan": "琼",
                "guizhou": "贵", "shanxi": "晋", "liaoning": "辽", "tianjin": "津", "gansu": "甘",
                "jilin": "吉", "neimenggu": "蒙", "xinjiang": "新", "ningxia": "宁", "xianggang": "港",
                "taiwan": "台", "qinghai": "青", "aomen": "澳", "xizang": "藏"}
province_abbr = [province_dict[i] for i in province]

report_date = np.array(df["date"])[::3]
duration = len(report_date)
report_date = [datetime.datetime.strptime(report_date[i], "%Y/%m/%d") for i in range(0, duration)]
forecast_date = [report_date[-1]+datetime.timedelta(days=i) for i in range(1, 8)]

confirmed_acc = np.array(df[df["cases"] == "total_confirmed"].iloc[:, 2:], dtype=np.int)
confirmed_delta = confirmed_acc[1:, :] - confirmed_acc[:-1, :]
recover_acc = np.array(df[df["cases"] == "total_recoveries"].iloc[:, 2:], dtype=np.int)
recover_delta = recover_acc[1:, :] - recover_acc[:-1, :]
death_acc = np.array(df[df["cases"] == "total_deaths"].iloc[:, 2:], dtype=np.int)
death_delta = death_acc[1:, :] - death_acc[:-1, :]

'''
# Exploratory analysis
c_res = sm.tsa.stattools.adfuller(x=confirmed_delta_sum) # p-value=0.00068
print("The p-value of the ADF unit root test is: ", c_res[1])
#sm.graphics.tsa.plot_pacf(x=confirmed_delta_sum)  
sm.graphics.tsa.plot_acf(x=confirmed_delta_sum)
plt.show()
'''

c_corr_mat = np.array([[lagged_cross_correlation(x=confirmed_delta[:, i], y=confirmed_delta[:, j], lag=1) for j in range(0, num_province)]
                       for i in range(0, num_province)])
r_corr_mat = np.array([[lagged_cross_correlation(x=recover_delta[:, i], y=recover_delta[:, j], lag=1) for j in range(0, num_province)]
                       for i in range(0, num_province)])

c_fit, c_fore, r_fit, r_fore = svr_forecast(c_delta=confirmed_delta, r_delta=recover_delta, forecast_length=7, with_province=True)
#c_fit, c_fore, r_fit, r_fore = corrsvr_forecast(c_delta=confirmed_delta, r_delta=recover_delta, c_ccmat=c_corr_mat, r_ccmat=r_corr_mat, forecast_length=7, with_province=True, kmax=15)
#c_fit, c_fore, r_fit, r_fore = xgboost_forecast(c_delta=confirmed_delta, r_delta=recover_delta, forecast_length=7)
#c_fit, c_fore, r_fit, r_fore = corrxgboost_forecast(c_delta=confirmed_delta, r_delta=recover_delta, c_ccmat=c_corr_mat, r_ccmat=r_corr_mat, forecast_length=7, kmax=15)
print(c_fore)
print(r_fore)

# [1] Plot the tendency
'''
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.plot(report_date[1:], np.sum(confirmed_delta, axis=1), color="firebrick", linestyle="-", label="new_confirmed")
ax.plot(report_date[1:], np.sum(recover_delta, axis=1), color="darkorange", linestyle="-", label="new_recoveries")
plt.gcf().autofmt_xdate()
ax.legend(fontsize='medium')
plt.savefig("tendency_plot.png", dpi=400)
'''

'''
# [2] Plot the correlation heatmap
lag_max = 3
fig, ax = plt.subplots(lag_max, 2, figsize=(20, 30))
for m in range(1, lag_max+1):
    c_corr_mat = np.array([[lagged_cross_correlation(x=confirmed_delta[:, i], y=confirmed_delta[:, j], lag=m) for j in range(0, num_province)]
                        for i in range(0, num_province)])
    r_corr_mat = np.array([[lagged_cross_correlation(x=recover_delta[:, i], y=recover_delta[:, j], lag=m) for j in range(0, num_province)]
                        for i in range(0, num_province)])

    ax0 = ax[m-1, 0].imshow(X=c_corr_mat, cmap="coolwarm", vmax=1.0, vmin=-1.0)
    ax[m-1, 0].set_xticks(np.arange(num_province))
    ax[m-1, 0].set_xticklabels(province_abbr, fontproperties=font_prop)
    ax[m-1, 0].set_yticks(np.arange(num_province))
    ax[m-1, 0].set_yticklabels(province_abbr, fontproperties=font_prop)
    ax[m-1, 0].set_title("Correlation for new_confirmed, lag="+str(m))
    fig.colorbar(ax0, ax=ax[m-1
    , 0])

    ax1 = ax[m-1, 1].imshow(X=r_corr_mat, cmap="coolwarm", vmax=1.0, vmin=-1.0)
    ax[m-1, 1].set_xticks(np.arange(num_province))
    ax[m-1, 1].set_xticklabels(province_abbr, fontproperties=font_prop)
    ax[m-1, 1].set_yticks(np.arange(num_province))
    ax[m-1, 1].set_yticklabels(province_abbr, fontproperties=font_prop)
    ax[m-1, 1].set_title("Correlation for new_recoveries, lag="+str(m))
    fig.colorbar(ax1, ax=ax[m-1, 1])
    
plt.savefig("corr_plot.png", dpi=400)
'''

# [3] Plot the result
c_true = np.array([517, 411, 440, 329, 430, 579, 206])
r_true = np.array([2596, 2422, 2756, 3626, 2892, 2626, 2844])
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.plot(report_date[1:], np.sum(confirmed_delta, axis=1), color="darkorange", linestyle="-", label="new_confirmed, Obs")
ax.plot(report_date[1:], np.sum(recover_delta, axis=1), color="c", linestyle="-", label="new_recoveries, Obs")
ax.plot(report_date[2:], c_fit, color="firebrick", linestyle="--", label="new_confirmed, Fit")
ax.plot(report_date[2:], r_fit, color="navy", linestyle="--", label="new_recoveries, Fit")
ax.plot(forecast_date, c_fore, color="firebrick", linestyle="-.", label="new_confirmed, Forecast")
ax.plot(forecast_date, r_fore, color="navy", linestyle="-.", label="new_recoveries, Forecast")
ax.scatter(forecast_date, c_true, color="darkorange", marker="^", label="new_confirmed, True")
ax.scatter(forecast_date, r_true, color="c", marker="^", label="new_recoveries, True")
plt.gcf().autofmt_xdate()
ax.legend(fontsize='medium')
plt.savefig("SVR_Res_plot_withProvince.png", dpi=400)

#plt.show()

