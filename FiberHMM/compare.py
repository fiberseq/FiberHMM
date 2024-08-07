import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests


l=pd.read_csv('Pol-II_dm6/Processed/bidirectional_1000_nondr_untreated.csv', index_col=0)
s=pd.read_csv('Pol-II_dm6/Processed/bidirectional_gene-names_fix_short-range_nondr_untreated.csv', sep='\t',index_col=0)

l=l.loc[l['dists']>250]

df=pd.concat([s,l])
gtf_df=pd.read_csv('Pol-II_dm6/Reference/dm6_info.csv', sep='\t')

genes=gtf_df.loc[gtf_df['PRO_TR']<10]['gid']

print(df)
s1=df.loc[df['g1'].isin(genes)]
s2=df.loc[df['g2'].isin(genes)]
s2.columns=['yy','ny','yn','nn','strand','dists','g2','g1']
s=pd.concat([s1,s2])
s=s.loc[s['yy']>0]
#plt.hist(np.log10((s['yn']+s['yy'])/(s['ny']+s['yy'])))
#plt.show()

bs=[]
cs=[]
for index, row in s.iterrows():
	total=row['yy']+row['yn']+row['ny']+row['nn']
	yes1=row['yy']+row['ny']
	yes2=row['yy']+row['yn']
	table=[[row['yy'],row['yn']],[row['ny'],row['nn']]]
	p=scipy.stats.fisher_exact(table)[1]
	if p<.1:
		bs.append(scipy.stats.binom_test([row['yy'], row['yn']], p=yes2/total, alternative='greater'))
		cs.append(scipy.stats.binom_test([row['yy'], row['ny']], p=yes1/total, alternative='greater'))

plt.hist(bs, alpha=.5)
plt.hist(cs, alpha=.5)
plt.show()



genes=gtf_df.loc[gtf_df['PRO_TR']>10]['gid']

print(df)
s1=df.loc[df['g1'].isin(genes)]
s2=df.loc[df['g2'].isin(genes)]
s2.columns=['yy','ny','yn','nn','strand','dists','g2','g1']
s=pd.concat([s1,s2])
s=s.loc[s['yy']>0]
#plt.hist(np.log10((s['yn']+s['yy'])/(s['ny']+s['yy'])))
#plt.show()

bs=[]
cs=[]
for index, row in s.iterrows():
	total=row['yy']+row['yn']+row['ny']+row['nn']
	yes1=row['yy']+row['ny']
	yes2=row['yy']+row['yn']
	table=[[row['yy'],row['yn']],[row['ny'],row['nn']]]
	p=scipy.stats.fisher_exact(table)[1]
	if p<.1:
		bs.append(scipy.stats.binom_test([row['yy'], row['yn']], p=yes2/total, alternative='greater'))
		cs.append(scipy.stats.binom_test([row['yy'], row['ny']], p=yes1/total, alternative='greater'))

plt.hist(bs, alpha=.5)
plt.hist(cs, alpha=.5)
plt.show()