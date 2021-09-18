# -*- coding: utf-8 -*-
# 获取特征贡献度
# %%
import pandas
from matplotlib import ticker

rollback = pandas.read_csv("./jdb_rollback.csv")

# 处理数据
# 还款标识
# rollback.loc[rollback['rollback_dt'] == "未还款", 'rollback_dt'] = 0
# rollback.loc[rollback['rollback_dt'] != "未还款", 'rollback_dt'] = 1
# 首次是否逾期
rollback.loc[rollback['首笔借款当前是否逾期'] != "是", '首笔借款当前是否逾期'] = 0
rollback.loc[rollback['首笔借款当前是否逾期'] == "是", '首笔借款当前是否逾期'] = 1
# 是否首次逾期
rollback.loc[rollback['是否首次逾期'] != "是", '是否首次逾期'] = 0
rollback.loc[rollback['是否首次逾期'] == "是", '是否首次逾期'] = 1

# 显示所有列
pandas.set_option('display.max_columns', None)
# 表头不换行
pandas.set_option('expand_frame_repr', False)
pandas.set_option('max_colwidth', 60)
print(rollback.loc[[811]])

# 利用feature_selection计算特征的重要程度
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt

# 在运行程序时出现 RuntimeWarning: invalid value encountered in true_divide的警告，可能是在使用numpy时出现了0除以0的情况造成的。
# 解决方法：
# np.seterr(divide='ignore', invalid='ignore')

# predictors = [
#     'overdue_days',
#     'overdue_amount',
#     'overdue_cnt',
#     'rate_min',
#     'rate_max',
#     'is_first_overdue_noback',
#     'is_first_overdue',
#     'borrow_cnt_with_delay',
#     'borrow_cnt',
#     'rollback_attime',
#     'rollback_overtime',
#     'call_14days',
#     'connect_14days',
#     'connect_keep_timimg'
# ]

predictors = [
    '逾期天数',
    '逾期金额',
    '逾期债务数',
    '最小利率',
    '最大利率',
    '首笔借款当前是否逾期',
    '是否首次逾期',
    '历史借款次数含展期',
    '历史借款次数不含展期',
    '历史非逾期还款次数',
    '历史逾期还款次数',
    # '前14天外呼数',
    # '前14天接通数',
    # '前14天接通时长'
]

selector = SelectKBest(f_classif, k=11)
selector.fit(rollback[predictors], rollback["rollback_dt"])
scores = -np.log(selector.pvalues_)

# 字体，解决汉字不显示问题
plt.rc("font", family='Heiti TC')
# 绘图
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# 画布偏移
plt.subplots_adjust(bottom=0.3)
# 横坐标旋转
plt.xticks(fontsize=8)
plt.show()

# %%
# 查询当前系统所有字体
# from matplotlib.font_manager import FontManager
#
# mpl_fonts = set(f.name for f in FontManager().ttflist)
#
# print('all font list get from matplotlib.font_manager:')
# for f in sorted(mpl_fonts):
#     print('\t' + f)
