import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
#reviewer : IDKullanıcı ID’si


df = pd.read_csv("measurement_problems/datasets/course_reviews.csv")
df.head()


df[(df["Progress"]>75)]["Rating"].mean()
df[(df["Progress"]<25)]["Rating"].mean()

test_stat, pvalue = shapiro(df[(df["Progress"]>75)]["Rating"])
print("test=%.4f, pvalue=%4.f" % (test_stat,pvalue))
test_stat, pvalue = shapiro(df[(df["Progress"]<25)]["Rating"])
print("test=%.4f, pvalue=%4.f" % (test_stat,pvalue))


test_stat, pvalue = mannwhitneyu(df[(df["Progress"]>75)]["Rating"],df[(df["Progress"]<25)]["Rating"])
print("test=%.4f, pvalue=%4.f" % (test_stat,pvalue))


#uygulama

basari_sayisi = np.array([300,250])
gozlem_sayisi = np.array([1000,1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayisi)


#titanic kadın erkek hayatta kalma oranları ist anl

df = sns.load_dataset("titanic")
df.head()
df.loc[df["sex"] =="female", "survived"].mean()
df.loc[df["sex"] == "male", "survived"].mean()


female_succ_count =df.loc[df["sex"] =="female", "survived"].sum()
male_succ_count = df.loc[df["sex"] =="male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count,male_succ_count],
                                      nobs=[df.loc[df["sex"] =="female", "survived"].shape[0],
                                      df.loc[df["sex"] =="male", "survived"].shape[0]])
print("test=%.4f, pvalue=%.4f" % (test_stat,pvalue))

###### ANOVA

df = sns.load_dataset("tips")
df.head()
df.groupby("day").agg({"total_bill":"mean"})


#parametrik mi non parametrik mi?

#H0 normallik?
for group in list(df["day"].unique()):
    pvalue= shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "pvalue: %.4f" % pvalue)

#Sun pvalue: 0.0036
#Sat pvalue: 0.0000
#Thur pvalue: 0.0000
#Fri pvalue: 0.0409
#h0 için normallik varsayımı sağlanmamaktadır.

#Ho varyans homojenliği?

test_stat, pvalue = levene(df.loc[df["day"] =="Sun", "total_bill"],
                           df.loc[df["day"] =="Sat", "total_bill"],
                           df.loc[df["day"] =="Thur", "total_bill"],
                           df.loc[df["day"] =="Fri", "total_bill"])
print("test : %.4f, pvalue: %.4f" % (test_stat,pvalue))
#test : 0.6654, pvalue: 0.5741
#reddedemiyoruz. varyans homojenliği sağlanıyor. normallikten düştük her türlü non parametrike gidecez


#3.Hipotez testi ve pvalue yorumu

#parametrik anova testi
f_oneway(df.loc[df["day"] =="Sun", "total_bill"],
         df.loc[df["day"] =="Sat", "total_bill"],
         df.loc[df["day"] =="Thur", "total_bill"],
         df.loc[df["day"] =="Fri", "total_bill"])


#non parametrik anova testi

kruskal(df.loc[df["day"] =="Sun", "total_bill"],
                           df.loc[df["day"] =="Sat", "total_bill"],
                           df.loc[df["day"] =="Thur", "total_bill"],
                           df.loc[df["day"] =="Fri", "total_bill"])
#KruskalResult(statistic=10.403076391436972, pvalue=0.015433008201042065)
#H0 reddedilir. anlamlı bir fark vardır. işlem tamamlandı

#İyi de fark kimden kaynaklanıyor???

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
#ikili karşılaştırmada anlamlı bir fark bulunamadı.






















