from icecream import ic
import numpy as np
import matplotlib as mpl

# mpl.use('Qt5Agg')
mpl.use('TkAgg')
import pandas as pd
from ThreeDTool import Up, Dspl, normalization

# with open("./plot_out/test_105.npy", 'rb') as f:
#     array = np.load(f)
# ic(array)
# plotter = Plotter("plot_out/test_105.npy")
# plotter.all_plot()
# plotter.xy_plot()
# plotter.xvx_t()
# plotter.reg_plot()
# plotter.dot_xy_plot()
# plotter.x_dot()
# plotter.y_dot()
# plotter.xy_pred()

# plotter2 = Plotter("plot_out/test_108_zeros.npy")
# # plotter2.all_plot()
# plotter2.xy_plot()
# # plotter2.xvx_t()
# # plotter2.reg_plot()
# # plotter2.dot_xy_plot()
# # plotter2.x_dot()
# # plotter2.y_dot()
# # plotter2.xy_pred()
def is_zero(arr):
    if np.allclose(arr, [0., 0., 0.], 1e-8):
        return True
    else:
        return False
df = pd.read_csv('./plot_out/data.csv', index_col=0)
df.plot(x='t')
# plt.show()
# ic(df[['x', 'y', 'z']].to_numpy())

up = Up(df[['x', 'y', 'z']], 'plot')
up_zeros = Up([[0, 0, 0]])
up_not_zeros = Up([[0, 0, 0]])


def zero_check(arr, i):
    if is_zero(arr[i]):
        ahead = 0
        behind = 0
        k = 1
        j = 1
        while True:
            # print(k, j)
            if i + k == arr.shape[0] - 1:
                break
            else:
                # if i + k == np.shape(arr)[0]-1:
                #     break
                ic(i+k-1)
                ic(arr.shape)
                if not is_zero(arr[i + k]):
                    break
            ahead += 1
            k += 1
            if i+k == np.shape(arr)[0]-100:
                break
        while True:
            if i - j == 0:
                break
            else:
                if not is_zero(arr[i - j]):
                    break
            behind += 1
            j += 1
        # print(f"ahead = {ahead}, behind = {behind}")
        P_ia1 = arr[i + ahead + 1]
        P_ib1 = arr[i - behind - 1]
        # print(f'Pa = {P_ia1}', f"Pb = {P_ib1}")
        var = P_ia1 - P_ib1
        L = np.linalg.norm(var)
        # print(f"L = {L}")
        v = normalization(var, L / (2 + ahead + behind))
        # print(f"v = {v}")
        out = P_ib1 + v * (behind + 1)
        return out
    else:
        return arr[i]


for i, obj in enumerate(up):
    if i == 0:
        continue
    elif i == up.shape[-2]:
        break
    if np.allclose(obj, [0., 0., 0.], 1e-8):
        point = zero_check(up, i)
        # ic(point)
        up_zeros = np.vstack([up_zeros, point])
    else:
        up_not_zeros = np.vstack([up_not_zeros, obj])


# def step_count()
# up = Up([[0, 0, 10000], [1, 1.5, 0.2], [0.4, 0.2, 1], [3, 3, 3], [20, 0, 0]])




up_zeros = Up(up_zeros)
up_not_zeros = Up(up_not_zeros, 'plot')
up_zeros.color = "red"
up_zeros.s = 50
# dp = Dspl([up_not_zeros, up_zeros])
# dp.show()

# Предположим, что у нас есть массивы t, x и y
# Преобразуем t в столбчатый вектор (n_samples, n_features)
# t = t.reshape(-1, 1)

# Создаем полиномиальные признаки
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=8)
x = df['x'].to_numpy().reshape((-1, 1))
x_poly = poly.fit_transform(x)

# Создаем модели линейной регрессии для полиномиальных признаков x и y
# model_x = LinearRegression()
model_y = LinearRegression()
model_z = LinearRegression()

# Обучаем модели

model_y.fit(x_poly, df['y'].to_numpy())
model_z.fit(x_poly, df['z'].to_numpy())

# Предсказываем значения
y_pred = model_y.predict(x_poly)
z_pred = model_z.predict(x_poly)
x = x.reshape(-1)
print(z_pred.shape, y_pred.shape, x.shape)

data = np.vstack([x, y_pred, z_pred])
print(data.T.shape)

up3 = Up(data.T)
up3.color = 'yellow'
dp = Dspl([up_not_zeros, up_zeros])
dp.show()
