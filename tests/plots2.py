from icecream import ic
import numpy as np
import matplotlib as mpl
from config import CONFIG

# mpl.use('Qt5Agg')
mpl.use('TkAgg')
import pandas as pd
from ThreeDTool import Up, Dspl, normalization
def is_zero(arr):
    if np.allclose(arr, [0., 0., 0.], 1e-8):
        return True
    else:
        return False
df = pd.read_csv('../plot_out/freq_count.csv', index_col=0)
print(df.head())
df = df.loc[1:]
print(df.head())
# df.plot(x='t', title=f'График позиционирования в точке. Частота = {CONFIG['period_get_xyz']}')
df_unique = df.drop_duplicates(subset=['x', 'y', 'z', 'Vx', 'Vy', 'Vz']).copy()
# df_unique.to_csv('./test.csv')

df_unique.loc[:, 'diff'] = df_unique['t'].diff()

# Вычисление средней разницы, исключая первое значение (так как оно будет NaN)
mean_diff = df_unique['diff'].mean()
df_unique.to_csv('./test.csv')
print(f"T = {mean_diff} \n freq = 1/T =  {1/mean_diff}")

#
# # Вычисляем разницу между максимальным и минимальным значением t для каждой группы
# df['t_diff'] = grouped['t'].transform(lambda x: x.max() - x.min())
#
# # Оставляем только уникальные группы
# result = df.drop_duplicates(subset=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
#
# # Среднее значение разницы t_diff
# mean_t_diff = result['t_diff'].mean()



import pandas as pd

# # Пример DataFrame
# data = {
#     'x': [1, 1, 2, 2, 1],
#     'y': [2, 2, 3, 3, 2],
#     'z': [3, 3, 4, 4, 3],
#     'Vx': [4, 4, 5, 5, 4],
#     'Vy': [5, 5, 6, 6, 5],
#     'Vz': [6, 6, 7, 7, 6],
#     't': [1, 3, 1, 4, 5]
# }
#
# import pandas as pd
#
# # Пример DataFrame
# data = {
#     'x': [1, 1, 2, 2, 1],
#     'y': [2, 2, 3, 3, 2],
#     'z': [3, 3, 4, 4, 3],
#     'Vx': [4, 4, 5, 5, 4],
#     'Vy': [5, 5, 6, 6, 5],
#     'Vz': [6, 6, 7, 7, 6],
#     't': [1, 3, 1, 4, 5]
# }
#
# df = pd.DataFrame(data)
#
# # Группировка по 'x', 'y', 'z', 'Vx', 'Vy', 'Vz' и расчет разницы max(t) - min(t)
# grouped = df.groupby(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
#
# # Вычисляем разницу между максимальным и минимальным значением t для каждой группы
# t_diffs = grouped['t'].agg(lambda x: x.max() - x.min())
#
# # Вычисление среднего значения разницы
# mean_t_diff = t_diffs.mean()
#
# # Вывод результата
# print("Средняя разница между начальным и конечным временем t по всем группам:", mean_t_diff)


# df = pd.DataFrame(data)
#
# # Группировка по 'x', 'y', 'z', 'Vx', 'Vy', 'Vz' и расчет разницы max(t) - min(t)
# grouped = df.groupby(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
#
# # Вычисляем разницу между максимальным и минимальным значением t для каждой группы
# t_diffs = grouped['t'].agg(lambda x: x.max() - x.min())
#
# # Вычисление среднего значения разницы
# mean_t_diff = t_diffs.mean()
#
# # Вывод результата
# print("Средняя разница между начальным и конечным временем t по всем группам:", mean_t_diff)
#
#
#
# # Вывод результатов
# print("Средняя разница между максимальным и минимальным значением t по всем группам:", mean_t_diff)
# print(result[['x', 'y', 'z', 'Vx', 'Vy', 'Vz', 't_diff']])

# # import pandas as pd
#
#
# # df = pd.DataFrame(data)
#
# # Группировка по 'x', 'y', 'z', 'Vx', 'Vy', 'Vz'
# grouped = df.groupby(['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
#
# # Вычисление разницы между максимальным и минимальным значением 't'
# df['t_diff'] = grouped['t'].transform(lambda x: x.max() - x.min())
#
# # Вычисление среднего значения времени 't' для каждой группы
# df['t_mean'] = grouped['t'].transform('mean')
#
# # Удаление дубликатов строк (оставляем только одну строку на группу)
# result = df.drop_duplicates(subset=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
#
# # Вывод результата
# print(result[['x', 'y', 'z', 'Vx', 'Vy', 'Vz', 't_diff', 't_mean']])

