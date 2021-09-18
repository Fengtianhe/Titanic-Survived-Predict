#! /bin/sh
# shellcheck disable=SC1072
# shellcheck disable=SC1020
# shellcheck disable=SC1073
# shellcheck disable=SC1035
# shellcheck disable=SC1069

# 不同系统取时间的方式不一致
# Mac OS X 操作系统
# yesterday=$(date -v -1d +%Y%m%d)
# GNU/Linux操作系统"
yesterday=$(date -d '1 days ago' +%Y%m%d)

echo "数据日期：$yesterday"

# 训练数据集
echo "获取训练数据集"
echo "hadoop fs -get \"/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/$yesterday/train.csv\" ./"
hadoop fs -get "/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/$yesterday/train.csv" ./
# 测试数据集
echo "获取测试数据集"
echo "hadoop fs -get \"/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/$yesterday/test.csv\" ./"
hadoop fs -get "/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/$yesterday/test.csv" ./

# python训练程序
echo "获取python训练程序"
echo "hadoop fs -get \"/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/application/train.py\" ./"
hadoop fs -get "/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention/application/train.py" ./

# 创建模型目录
echo "创建模型目录"
mkdir model

echo "执行python程序"
python train.py