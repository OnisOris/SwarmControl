import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')


class Plotter:
    def __init__(self, data_path: str, columns=None, title: str = "Plots", out_img_path: str = "./"):
        if columns is None:
            self.columns = ['x', 'y', 'z', 'vx', 'vy', 't']
        with open(data_path, 'rb') as f:
            self.array = np.load(f)
        self.df = pd.DataFrame(self.array, columns=self.columns)
        self.out_img_path = out_img_path
        self.title = title
        self.data_path = data_path
        self.plot = True
        self.x_pred = None
        self.y_pred = None

    def all_plot(self):
        if self.plot:
            self.df.plot(x='t', title=self.title)
            plt.show()
    def xy_plot(self):
        if self.plot:
            coord = self.df[['x', 'y']]
            coord.plot(x='x', y='y', title=self.title)
            plt.show()
    def xvx_t(self):
        if self.plot:
            s = self.df[['x', 'vx', 't']]
            s.plot(x='t', title=self.title)
            plt.show()
    def reg_plot(self, plot=True):

        # Предположим, что у нас есть массивы t, x и y
        t = self.array[:, 5]
        # t = t.reshape(-1, 1)
        x = self.array[:, 0]
        y = self.array[:, 1]

        # Преобразуем t в столбчатый вектор (n_samples, n_features)
        t = t.reshape(-1, 1)

        # Создаем полиномиальные признаки
        poly = PolynomialFeatures(degree=8)
        t_poly = poly.fit_transform(t)

        # Создаем модели линейной регрессии для полиномиальных признаков x и y
        model_x = LinearRegression()
        model_y = LinearRegression()

        # Обучаем модели
        model_x.fit(t_poly, x)
        model_y.fit(t_poly, y)

        # Предсказываем значения
        self.x_pred = model_x.predict(t_poly)
        self.y_pred = model_y.predict(t_poly)

        # Выводим коэффициенты регрессии
        if self.plot and plot:
            print("Коэффициенты полиномиальной регрессии для x:")
            print("Коэффициенты:", model_x.coef_)
            print("Свободный член:", model_x.intercept_)

            print("Коэффициенты полиномиальной регрессии для y:")
            print("Коэффициенты:", model_y.coef_)
            print("Свободный член:", model_y.intercept_)

            # Визуализируем результаты
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.scatter(t, x, color='blue', label='Исходные данные x')
            plt.plot(t, self.x_pred, color='red', label='Полиномиальная регрессия x')
            plt.title('Полиномиальная регрессия для x')
            plt.xlabel('Время (t)')
            plt.ylabel('Координата (x)')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(t, y, color='green', label='Исходные данные y')
            plt.plot(t, self.y_pred, color='orange', label='Полиномиальная регрессия y')
            plt.title('Полиномиальная регрессия для y')
            plt.xlabel('Время (t)')
            plt.ylabel('Координата (y)')
            plt.legend()

            plt.tight_layout()
            plt.show()

    def dot_xy_plot(self):
        if self.x_pred is None:
            self.reg_plot(plot=False)
        tu = self.array[:, 5]
        x_dot2 = np.gradient(self.x_pred, tu)
        y_dot2 = np.gradient(self.y_pred, tu)
        df_dot2 = pd.DataFrame(np.vstack([x_dot2, y_dot2, self.array[:, 3], self.array[:, 4], tu]).T,
                               columns=['x_dot', 'y_dot', 'vx', 'vy', 't'])
        df_dot2.plot(x='t', title=self.title)
        plt.show()

    def x_dot(self):
        if self.x_pred is None:
            self.reg_plot(plot=False)
        tu = self.array[:, 5]
        x_dot2 = np.gradient(self.x_pred, tu)
        y_dot2 = np.gradient(self.y_pred, tu)
        df_dot2 = pd.DataFrame(np.vstack([x_dot2, tu]).T, columns=['x_dot', 't'])
        df_dot2.plot(x='t', title=self.title)
        plt.show()

    def y_dot(self):
        if self.x_pred is None:
            self.reg_plot(plot=False)
        tu = self.array[:, 5]
        y_dot2 = np.gradient(self.y_pred, tu)
        df_dot2 = pd.DataFrame(np.vstack([y_dot2, self.array[:, 4], tu]).T, columns=['y_dot', 'vy', 't'])
        df_dot2.plot(x='t', title=self.title)
        plt.show()

    def xy_pred(self):
        if self.x_pred is None:
            self.reg_plot(plot=False)
        xy_pred = pd.DataFrame(np.vstack([self.x_pred, self.y_pred]).T)
        xy_pred.plot(x=0, title=self.title)
        plt.show()
