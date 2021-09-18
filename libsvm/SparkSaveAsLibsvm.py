# -*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

if __name__ == '__main__':
    sc = SparkContext()
    hc = HiveContext(sc)

    df = hc.sql("select product_key,plu_id "
                "from hdp_lbg_zhaopin_defaultdb.dws_zp_buser_zptag_areacate_result "
                "where dt = 20210905")

    d = df.rdd.map(lambda line: LabeledPoint(line[0], [str(a) for a in line[1:]]))

    hdfsPath = 'viewfs://58-cluster/home/hdp_lbg_zhaopin/resultdata/beehive_busertag/consumption_intention'
    MLUtils.saveAsLibSVMFile(d, hdfsPath)


