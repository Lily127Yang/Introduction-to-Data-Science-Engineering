# 酒店预定需求分析&取消率预测
# 探索性分析部分代码
import pandas as pd
import numpy as np

# 数据分析&绘图
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings

warnings.filterwarnings("ignore")

# 科学计算
from scipy.stats import skew, kurtosis
import pylab as py

# 时间
import time
import datetime
from datetime import datetime
from datetime import date
import calendar

data = pd.read_csv('data.csv')
data.info()
# 查看缺失数据
print(data.isnull().sum()[data.isnull().sum() != 0])
# 缺失值处理
# company 缺失太多，删除
# country、children和agent缺失比较少，用字段内的众数填充
# country和children用字段内的众数填充 agent缺失值用0填充，代表没有指定任何机构
data_new = data.copy(deep=True)
data_new.drop("company", axis=1, inplace=True)
data_new["agent"].fillna(0, inplace=True)
data_new["children"].fillna(data_new["children"].mode()[0], inplace=True)
data_new["country"].fillna(data_new["country"].mode()[0], inplace=True)
print(data_new.isnull().sum()[data_new.isnull().sum() != 0])
print(data_new.head(10))
print(data_new[['children', 'agent', 'country']])
data_new.info()
print(data_new.isnull().sum()[data_new.isnull().sum() != 0])

# 处理一下异常值：成人+小孩+婴儿=0的情况都需要删掉
data_new["children"] = data_new["children"].astype(int)
data_new["agent"] = data_new["agent"].astype(int)  # 转换数据类型
zero_guests = list(data_new["adults"] + data_new["children"] + data_new["babies"] == 0)
data_new.drop(data_new.index[zero_guests], inplace=True)
# meal字段映射处理
data_new["meal"].replace(["Undefined", "BB", "FB", "HB", "SC"],
                         ["No Meal", "Breakfast", "Full Board", "Half Board", "No Meal"], inplace=True)
# 数据去重
data_new.drop_duplicates(inplace=True)
data_new.to_csv('data_new.csv')
print(data.shape)
print(data_new.shape)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 我们对 城市酒店 和 度假酒店 进行统计分析
# 酒店预定情况
# 总体概况
labels = ['City Hotel', 'Resort Hotel']
colors = ["#538B8B", "#7AC5CD"]
order = data_new['hotel'].value_counts().index
plt.figure(figsize=(19, 9))
plt.suptitle("酒店预定情况", fontweight='heavy', fontsize='25', fontfamily='sans-serif', color="black")
# 饼图
plt.subplot(1, 2, 1)
plt.title('饼图', fontweight='bold', fontsize=20, fontfamily="sans-serif", color='black')
plt.pie(data_new["hotel"].value_counts(), pctdistance=0.7, autopct='%.2f%%', labels=labels,
        wedgeprops=dict(alpha=0.8, edgecolor="black"), textprops={'fontsize': 20}, colors=colors)
centre = plt.Circle((0, 0), 0.45, fc="white", edgecolor="black")
plt.gcf().gca().add_artist(centre)
# 柱状图
countplt = plt.subplot(1, 2, 2)
plt.title("柱状图", fontweight="bold", fontsize=20, fontfamily="sans-serif", color='black')
ax = sns.countplot(x="hotel", data=data_new, order=order, edgecolor="black", palette=colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 4.25, rect.get_height(),
            horizontalalignment="center", fontsize=10,
            bbox=dict(facecolor="none", edgecolor="black", linewidth=0.25, boxstyle="round"))
plt.xlabel("Hotel", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.ylabel("Number Of Bookings", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.xticks([0, 1], labels)
plt.grid(axis="y", alpha=0.4)
data_new['hotel'].value_counts()
plt.show()

# 每年不同酒店的预订量
df_ = data_new.groupby(['arrival_date_year', 'hotel'])['is_canceled'].agg(
    [('count', 'count'), ('canceled', sum)]).reset_index()
df_ = df_.groupby(['arrival_date_year', 'hotel'])['count', 'canceled'].mean().reset_index()
df_['canceled_rate'] = df_['canceled'] / df_['count']
df_['check_in'] = df_['count'] - df_['canceled']
fig = plt.figure(figsize=(10, 15))
ax1 = fig.add_subplot(1, 1, 1)
sns.barplot(data=df_, x='arrival_date_year', y='count', hue='hotel', palette='Blues', ax=ax1)
ax1.set_title('每年不同酒店的预订量', fontsize=16)
plt.show()

# 订单取消情况分析
labels = ['0：未取消', '1：已取消']
colors = ["#538B8B", "#7AC5CD"]
order = data_new['is_canceled'].value_counts().index
plt.figure(figsize=(19, 9))
plt.suptitle('订单取消情况分析', fontweight='heavy', fontsize=20, fontfamily='sans-serif', color='black')
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('饼图', fontweight='bold', fontsize=20, fontfamily="sans-serif", color='black')
plt.pie(data_new["is_canceled"].value_counts(), pctdistance=0.7, autopct='%.2f%%', labels=labels,
        wedgeprops=dict(alpha=0.8, edgecolor="black"), textprops={'fontsize': 20}, colors=colors)
centre = plt.Circle((0, 0), 0.45, fc="white", edgecolor="black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("柱状图", fontweight="bold", fontsize=20, fontfamily="sans-serif", color='black')
ax = sns.countplot(x="is_canceled", data=data_new, order=order, edgecolor="black", palette=colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 4.25, rect.get_height(),
            horizontalalignment="center", fontsize=10,
            bbox=dict(facecolor="none", edgecolor="black", linewidth=0.25, boxstyle="round"))
plt.xlabel("订单取消与否", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.ylabel("订单数", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.xticks([0, 1], labels)
plt.grid(axis="y", alpha=0.4)
plt.show()
print(data_new.is_canceled.value_counts())

# 两种酒店的订单情况
df_ = data_new.groupby(['hotel', 'is_canceled'])['lead_time'].count()


def func(pct, data):
    b = int(pct * data / 100.)
    return "{:.2f}%\n{:d}".format(pct, b)


fig = plt.figure(figsize=(19, 9))
cmap = plt.get_cmap("Pastel1")
outer_colors = cmap(np.arange(2) * 4)
inner_colors = cmap([1, 2, 5, 6])
ax1 = fig.add_subplot(1, 1, 1)
w1, t1, at1 = ax1.pie([df_.loc['City Hotel'].sum(), df_.loc['Resort Hotel'].sum()], radius=1,
                      wedgeprops=dict(width=0.4, edgecolor='w'), pctdistance=0.75, colors=outer_colors,
                      autopct=lambda pct: func(pct, df_.sum()))
w2, t2, at2 = ax1.pie(df_, radius=0.6, wedgeprops=dict(width=0.4, edgecolor='w'), colors=inner_colors,
                      autopct=lambda pct: func(pct, df_.sum()))
ax1.set_title('两种酒店的订单情况', fontsize=20)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.42)
kw = dict(arrowprops=dict(arrowstyle="-", color="k"), bbox=bbox_props, zorder=0.1, va="center")
for i, p in enumerate(w2):
    text = ["城市酒店-未取消", "城市酒店-取消", "度假酒店-未取消", "度假酒店-取消"]
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle, angleA=0, angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    plt.annotate(text[i], xy=(x, y), xytext=(1.15 * np.sign(x), 1.2 * y), horizontalalignment=horizontalalignment, **kw,
                 fontsize=20)
    plt.legend(w1, ["城市酒店", "度假酒店"], loc="upper right", bbox_to_anchor=(0, 0, 0.2, 1), fontsize=20)
plt.show()


# 顾客背景
# 新顾客中取消分布&老顾客中取消分布
def func(pct, data):
    b = int(pct * data / 100.)
    return "{:.2f}%\n{:d}".format(pct, b)


df_ = data_new.groupby(['is_repeated_guest'])['is_canceled'].agg([('count', 'count'), ('canceled', sum)])
df_['checkin'] = df_['count'] - df_['canceled']
colors = ["#538B8B", "#7AC5CD"]
fig = plt.figure(figsize=(19, 9))
labels = ['canceled', 'checkin']
ax1 = fig.add_subplot(1, 2, 1)
w, a, t = ax1.pie(df_.loc[0][['canceled', 'checkin']], colors=colors, labels=labels, shadow=True,
                  textprops=dict(color='black'), autopct=lambda pct: func(pct, df_.loc[0]['count']))
ax1.set_title('新顾客中取消分布', fontsize=16)
ax1.legend(w, labels)
ax2 = fig.add_subplot(1, 2, 2)
ax2.pie(df_.loc[1][['canceled', 'checkin']], colors=colors, labels=labels, shadow=True, textprops=dict(color='black'),
        autopct=lambda pct: func(pct, df_.loc[1]['count']))
ax2.set_title('老顾客中取消分布', fontsize=16)
ax2.legend(w, labels)
plt.show()

# 不同种类顾客的预定情况
labels = ['新顾客', '回头客']
colors = ["#538B8B", "#7AC5CD"]
order = data_new['is_repeated_guest'].value_counts().index
plt.figure(figsize=(19, 9))
plt.suptitle('不同种类顾客的预定情况', fontweight='heavy', fontsize='20', fontfamily='sans-serif', color="black")
# 饼饼
plt.subplot(1, 2, 1)
plt.title('饼图', fontweight='bold', fontsize=20, fontfamily="sans-serif", color='black')
plt.pie(data_new["is_repeated_guest"].value_counts(), pctdistance=0.7, autopct='%.2f%%', labels=labels,
        wedgeprops=dict(alpha=0.8, edgecolor="black"), textprops={'fontsize': 20}, colors=colors)
centre = plt.Circle((0, 0), 0.45, fc="white", edgecolor="black")
plt.gcf().gca().add_artist(centre)
# 柱柱
countplt = plt.subplot(1, 2, 2)
plt.title("柱状图", fontweight="bold", fontsize=20, fontfamily="sans-serif", color='black')
ax = sns.countplot(x="is_repeated_guest", data=data_new, order=order, edgecolor="black", palette=colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 4.25, rect.get_height(),
            horizontalalignment="center", fontsize=10,
            bbox=dict(facecolor="none", edgecolor="black", linewidth=0.25, boxstyle="round"))
plt.xlabel("顾客种类", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.ylabel("Total", fontweight="bold", fontsize=20, fontfamily="sans-serif", color="black")
plt.xticks([0, 1], labels)
plt.grid(axis="y", alpha=0.4)
plt.show()

# 客户分布图（世界地图版）
df_ = data_new.groupby(['hotel', 'country'])['is_canceled'].agg([('count', 'count')]).reset_index()
map = px.choropleth(df_, locations="country", color="count", hover_name="country",
                    color_continuous_scale=px.colors.sequential.Plasma, title="游客分布")
map.show()

# 酒店经营状况
# 不同月份平均旅客数
rh_bookings_monthly = data_new[data_new.hotel == "Resort Hotel"].groupby("arrival_date_month")["hotel"].count()
ch_bookings_monthly = data_new[data_new.hotel == "City Hotel"].groupby("arrival_date_month")["hotel"].count()

rh_bookings_data = pd.DataFrame({"arrival_date_month": list(rh_bookings_monthly.index), "hotel": "度假酒店",
                                 "guests": list(rh_bookings_monthly.values)})
ch_bookings_data = pd.DataFrame({"arrival_date_month": list(ch_bookings_monthly.index), "hotel": "城市酒店",
                                 "guests": list(ch_bookings_monthly.values)})
full_booking_monthly_data = pd.concat([rh_bookings_data, ch_bookings_data], ignore_index=True)

ordered_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
month_che = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"]

for en, che in zip(ordered_months, month_che):
    full_booking_monthly_data["arrival_date_month"].replace(en, che, inplace=True)

full_booking_monthly_data["arrival_date_month"] = pd.Categorical(full_booking_monthly_data["arrival_date_month"],
                                                                 categories=month_che, ordered=True)
full_booking_monthly_data.loc[(full_booking_monthly_data["arrival_date_month"] == "七月") | (
            full_booking_monthly_data["arrival_date_month"] == "八月"), "guests"] /= 3
full_booking_monthly_data.loc[~((full_booking_monthly_data["arrival_date_month"] == "七月") | (
            full_booking_monthly_data["arrival_date_month"] == "八月")), "guests"] /= 2
plt.figure(figsize=(12, 8))
sns.lineplot(x="arrival_date_month", y="guests", hue="hotel", hue_order=["城市酒店", "度假酒店"],
             data=full_booking_monthly_data, size="hotel", sizes=(2.5, 2.5))
plt.title("不同月份平均旅客数", fontsize=16)
plt.xlabel("月份", fontsize=16)
plt.ylabel("旅客数", fontsize=16)
plt.show()

# 每月不同酒店的预订量、入住量和取消率
data_new['arrival_date_month'] = data_new['arrival_date_month'].map((list(calendar.month_name)).index)
df_ = data_new.groupby(['arrival_date_year', 'arrival_date_month', 'hotel'])['is_canceled'].agg(
    [('count', 'count'), ('canceled', sum)]).reset_index()
df_ = df_.groupby(['arrival_date_month', 'hotel'])['count', 'canceled'].mean().reset_index()
df_['canceled_rate'] = df_['canceled'] / df_['count']
df_['check_in'] = df_['count'] - df_['canceled']
fig = plt.figure(figsize=(10, 15))
# 每月不同酒店的预订量
ax1 = fig.add_subplot(1, 1, 1)
sns.barplot(data=df_, x='arrival_date_month', y='count', hue='hotel', palette='Blues', ax=ax1)
ax1.set_title('每月不同酒店的预订量', fontsize=16)
plt.show()
# 每月不同酒店的入住量
ax2 = fig.add_subplot(1, 1, 1)
sns.barplot(data=df_, x='arrival_date_month', y='check_in', hue='hotel', palette='Blues', ax=ax2)
ax2.set_title('每月不同酒店的入住量', fontsize=16)
plt.show()
# 每月不同酒店的取消率
ax3 = fig.add_subplot(1, 1, 1)
sns.lineplot(data=df_, x='arrival_date_month', y='canceled_rate', hue='hotel', palette='Blues', ax=ax3)
ax3.set_title('每月不同酒店的取消率', fontsize=16)
plt.show()

# 不同月份人均居住价格/晚
data_new["adr_pp"] = data_new["adr"] / (data_new["adults"] + data_new["children"])
full_data_guests = data_new.loc[data_new["is_canceled"] == 0]  # only actual gusts
room_prices = full_data_guests[["hotel", "reserved_room_type", "adr_pp"]].sort_values("reserved_room_type")
room_price_monthly = full_data_guests[["hotel", "arrival_date_month", "adr_pp"]].sort_values("arrival_date_month")
ordered_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
month_che = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月", ]

for en, che in zip(ordered_months, month_che):
    room_price_monthly["arrival_date_month"].replace(en, che, inplace=True)
room_price_monthly["arrival_date_month"] = pd.Categorical(room_price_monthly["arrival_date_month"],
                                                          categories=month_che, ordered=True)
room_price_monthly["hotel"].replace("City Hotel", "城市酒店", inplace=True)
room_price_monthly["hotel"].replace("Resort Hotel", "度假酒店", inplace=True)
plt.figure(figsize=(12, 8))
sns.lineplot(x="arrival_date_month", y="adr_pp", hue="hotel", data=room_price_monthly, hue_order=["城市酒店", "度假酒店"],
             size="hotel", sizes=(2.5, 2.5))
plt.title("不同月份人均居住价格/晚", fontsize=16)
plt.xlabel("月份", fontsize=16)
plt.ylabel("人均居住价格/晚", fontsize=16)
plt.show()

# 顾客行为分析-天数选择
color = ['#ec7e32', '#a5a5a5']
data_new['total_adr'] = (data_new['stays_in_weekend_nights'] + data_new['stays_in_week_nights']) * data_new['adr']
data_new.pivot_table(values='total_adr', index='arrival_date_year', columns='hotel', aggfunc='sum').plot.bar()
plt.show()
plt.figure(figsize=(19, 19))
data_new["total_nights"] = data_new["stays_in_weekend_nights"] + data_new["stays_in_week_nights"]
order = data_new.total_nights.value_counts().iloc[:10].index
plt.title("顾客in不同酒店的总天数", fontweight="bold", fontsize=20, fontfamily="sans-serif", color='black')
ax = sns.countplot(x="total_nights", data=data_new, hue="hotel", edgecolor="black", palette="tab20c", order=order)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 4.25, rect.get_height(),
            horizontalalignment="center", fontsize=14,
            bbox=dict(facecolor="none", edgecolor="black", linewidth=0.25, boxstyle="round"))
plt.xlabel("天数", fontweight="bold", fontsize=16, fontfamily="sans-serif", color="black")
plt.ylabel("订单量", fontweight="bold", fontsize=15, fontfamily="sans-serif", color="black")
plt.grid(axis="y", alpha=0.4)
plt.show()

# 顾客行为分析-特殊要求
# 不同类型顾客的平均特殊要求量
df2 = data_new.groupby("customer_type")["total_of_special_requests"].mean().sort_values(ascending=False)[: 20]
plt.figure(figsize=(15, 18))
sns.barplot(x=df2.index, y=df2, palette='Blues')
plt.xticks(rotation=30)
plt.xlabel("客户类型", fontsize=16)
plt.ylabel("平均特殊要求量", fontsize=16)
plt.title("不同顾客类型的平均特殊要求量", fontweight="bold", fontsize=20, fontfamily="sans-serif", color='black')
plt.show()

# 各月的特殊要求数
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
df2 = data_new.groupby("arrival_date_month")["total_of_special_requests"].mean().sort_values(ascending=False)[: 20]
plt.figure(figsize=(18, 8))
sns.barplot(x=df2.index, y=df2, order=months, palette='Blues')
plt.xticks(rotation=30)
plt.xlabel("月份")
plt.ylabel("平均特殊要求数")
plt.title("各月的特殊要求数 ", fontweight="bold", fontsize=14, fontfamily="sans-serif", color='black')
plt.show()

# 顾客行为分析-渠道选择
# 预定的不同渠道
plt.figure(figsize=(19, 9))
b = data_new.groupby(by='market_segment').count()['hotel'].sort_values(ascending=False)
b = b[b.values > 500]
plt.title('预定的不同渠道', loc='right', fontweight='bold', fontsize=20, fontfamily="sans-serif", color='black')
plt.pie(b, labels=b.index, explode=[0, 0, 0.1, 0.1, 0.15, 0.2],
        colors=['lightblue', 'skyblue', 'lightskyblue', 'steelblue', 'powderblue', 'deepskyblue'], counterclock=False,
        autopct='%.1f%%', startangle=90, labeldistance=1.2)
plt.legend()
plt.show()

# 不同预定渠道的取消量
plt.figure(figsize=(10, 6))
sns.countplot(x='market_segment', hue='is_canceled', data=data_new, palette='Set3')
plt.title('不同预定渠道的取消量', fontsize=20)
plt.xticks(rotation=30)
plt.show()

# 顾客行为&取消率-历史取消订单量
# 历史订单取消率分布
# 复制df_copy表
df_copy = data_new.copy()
# 计算历史订单取消率，创建新列'canceled_rate'
sns.set(style="darkgrid")
df_copy['canceled_rate'] = (df_copy['previous_cancellations'] / (
            df_copy['previous_bookings_not_canceled'] + df_copy['previous_cancellations'])).replace(np.nan, 0)
grouped_df_copy_pb = df_copy.pivot_table(values='is_canceled', index='canceled_rate', aggfunc='mean')
# 可视化
sns.jointplot(x=grouped_df_copy_pb.index, y=grouped_df_copy_pb.is_canceled, kind='reg', size=12)
plt.title('pre_cancel&canceled_rate', fontsize=10)
plt.show()

# 顾客行为&取消率-提前预定时长
# lead_cancel_data
# 因为lead_time中值范围大且数量分布不匀，所以选取lead_time>10次的数据（<10的数据不具代表性）
booking_changes_data = pd.DataFrame(data_new.groupby("booking_changes")["is_canceled"].describe())
booking_changes_new = booking_changes_data[booking_changes_data["count"] < 10]
y = list(round(booking_changes_new["mean"], 4) * 100)
y = list(round(booking_changes_new["mean"], 4) * 100)
plt.figure(figsize=(19, 9))
sns.regplot(x=list(booking_changes_new.index), y=y, color='green')
plt.title("订单改动与取消率的相关性", fontsize=16)
plt.xlabel("订单改动", fontsize=16)
plt.ylabel("取消率(%)", fontsize=16)
plt.show()

# 顾客行为&取消率-房型匹配率
# 复制df_copy表
df_copy_rt = data_new.copy()
# 创建字段'room_type_agreed',判断预订房型与酒店分配房型是否一致
df_copy_rt['room_type_agreed'] = np.where(df_copy_rt['reserved_room_type'] == df_copy_rt['assigned_room_type'], 'yes',
                                          'no')
# 房型是否一致的订单量
grouped_df_rt = df_copy_rt.pivot_table(values='hotel', index='room_type_agreed', columns='is_canceled', aggfunc='count')
# 可视化
grouped_df_rt.plot(kind='bar', title='房型是否匹配&是否取消订单', fontsize=16, color=['lightblue', 'skyblue'])
plt.show()

# 平均每日房价&取消率
adr_cancel_data = data_new.groupby("adr")["is_canceled"].describe()
adr_cancel_data_10 = adr_cancel_data.loc[adr_cancel_data["count"] > 10]
plt.figure(figsize=(19, 9))
x, y = pd.Series(adr_cancel_data_10.index, name="x_var"), pd.Series(adr_cancel_data_10["mean"].values * 100,
                                                                    name="y_var")
sns.regplot(x=x, y=adr_cancel_data_10["mean"].values * 100, color='green')
plt.title("Effect of adr on cancelation", fontsize=16)
plt.xlabel("adr", fontsize=16)
plt.ylabel("取消率[%]", fontsize=16)
plt.show()

# 相关性分析
# 相关系数矩阵
corr_matrix = round(data_new.corr(), 3)
"Correlation Matrix: "
corr_matrix.to_csv('correlation.csv')
print(corr_matrix)
# 相关系数
cancel_corr = data_new.corr()["is_canceled"]
print(cancel_corr.abs().sort_values(ascending=False)[1:])
plt.rcParams['figure.figsize'] = (30, 30)
sns.heatmap(data_new.corr(), annot=True, cmap="YlGnBu", linewidths=5)
plt.suptitle('Correlation Between Variables', fontweight='heavy', x=0.03, y=0.98, ha="left", fontsize='18',
             fontfamily='sans-serif', color="black")
plt.show()

# 计算各个特征与is_canceled相关系数
cancel_corr = data_new.corr()["is_canceled"]
print(cancel_corr.abs().sort_values(ascending=False))
