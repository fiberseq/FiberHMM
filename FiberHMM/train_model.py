import pandas as pd
import numpy as np
import getopt
import sys
from hmmlearn import hmm
import os
import pickle

def options():
	#default parameters
	count = 10
	read_count = 150
	random_seed = 1



	optlist, args = getopt.getopt(sys.argv[1:],'i:f:c:r:s:e:',
		['indir=','infiles=','count=','read_count=','random_seed=', 'context='])

	for o, a in optlist:
		if o=='-i' or o=='--indir':
			indir=a
		elif o=='-f' or o=='--infiles':
			infiles=a.split(',')
		elif o=='-c' or o=='--count':
			count=int(a)
		elif o=='-r' or o=='--read_count':
			read_count=int(a)
		elif o=='-s' or o=='--random_seed':
			random_seed=int(a)	
		elif o=='-e' or o=='--context':
			context=a	
	return indir, infiles, count, read_count, random_seed, context

def make_emission_probs(acc_in, inacc_in):
	#generate dictionary of all hexamers
	bases=['A','C','T','G']
	trimers=[]
	for i in range(4):
		for j in range(4):
			for k in range(4):
				trimers.append(bases[i]+bases[j]+bases[k])
	hexamers=[]
	for i in range(len(trimers)):
		for j in range(len(trimers)):
			hexamers.append(trimers[i]+'A'+trimers[j])

	hexamers=dict(zip(hexamers,range(len(hexamers))))

	#import accessible/inaccessible probabilities merge into dataframe
	acc=pd.read_csv(acc_in, sep='\t', usecols=[0,3])
	acc.columns=['encode', 'prob_acc']
	inacc=pd.read_csv(inacc_in, sep='\t',usecols=[0,3])
	inacc.columns=['encode', 'prob_inacc']

	acc['encode']=acc['encode'].astype(int)
	acc=acc.sort_values(by='encode').reset_index()
	acc['prob_acc']=acc['prob_acc'].astype(float)
	inacc['encode']=inacc['encode'].astype(int)
	inacc=inacc.sort_values(by='encode').reset_index()
	inacc['prob_inacc']=inacc['prob_inacc'].astype(float)
	hexamer_probs=acc.merge(inacc)
	checks=np.arange(4097)
	missing=np.setxor1d(hexamer_probs['encode'].to_numpy(), checks)
	missing=pd.DataFrame([missing,[0]*len(missing), [0]*len(missing)]).T
	missing.columns=['encode','prob_acc','prob_inacc']
	hexamer_probs=pd.concat([hexamer_probs,missing])
	hexamer_probs=hexamer_probs.sort_values(by='encode')
	
	emission_probs=[hexamer_probs['prob_acc'].to_list()+(1-hexamer_probs['prob_acc']).to_list(),
					hexamer_probs['prob_inacc'].to_list()+(1-hexamer_probs['prob_inacc']).to_list()]

	return emission_probs

def encode_me(rid, read, read_info, context):

	chrom=read_info.loc[rid]['chrom']
	me=np.array(read.dropna())
	me=me[1:-1]
	me=me.astype(int)
	start=read_info.loc[rid,'start']
	end=read_info.loc[rid,'end']
	me=me[np.where(me<(end-start))[0]]
	no_me=np.arange(end-start)
	no_me=np.delete(no_me, me)
	hexamers=pd.read_hdf(context, key=chrom, start=start, stop=end)
	#make sure the methylations are within the correct range
	me=me[np.where(me<=len(hexamers))[0]]
	no_me=no_me[np.where(no_me<=len(hexamers))[0]]
	me_encode=hexamers.to_numpy().T[0]
	no_me_encode=me_encode+4097
	me_encode[no_me]=0
	no_me_encode[me]=0
	return me_encode+no_me_encode


def generate_training(indir, infiles, count, read_count, random_seed, context):

	train_df=pd.DataFrame()
	read_count=read_count//len(infiles)

	train_rids=pd.DataFrame()
	print(read_count)
	for f in infiles:
		print(f)
		indir_f=indir+'/Infiles/parquet/'+f+'/'
		read_info=pd.read_parquet(indir_f+f+'_read-info.pq')
		#print(read_info)
		train_reads=read_info.sample(n=read_count, random_state=random_seed)
		print(train_reads)
		
		#saving list of reads used in training
		ds=[f]*train_reads.shape[0]
		tmp = pd.DataFrame(ds)
		tmp.index=train_reads.index
		tmp['read_num']=tmp.index
		tmp=tmp.reset_index()
		tmp=tmp.drop(columns='index')
		tmp.columns=['dataset','read_num']
		train_rids=pd.concat([train_rids,tmp])

		print('randomly selecting reads')
		reads=pd.DataFrame()
		for chrom in train_reads['chrom'].unique():
			#if chrom!='chrM':
			tmp=pd.read_parquet(indir_f+f+'_'+chrom+'.pq', columns=train_reads.loc[train_reads['chrom']==chrom].index)
			reads=pd.concat([reads,tmp.T])
	
		print('encoding methylation')
		train_dic={}    
		for rid, read in reads.iterrows():
			encode=encode_me(rid, read, read_info, context)
			train_dic[rid]=encode

		tmp=pd.DataFrame.from_dict(train_dic, orient='index')
		train_df=pd.concat([train_df,tmp])

	print('generating training array')
	train_arrays={}
	for i in range(count):
		train_df=train_df.sample(frac=1, random_state=i)
		train_array=train_df.to_numpy().flatten()
		train_array = train_array[~np.isnan(train_array)].astype(int)
		train_arrays[i]=train_array

	return train_arrays, train_rids

def trained_HMM(emission_probs, train_arrays):
	#train HMM using dataset
	print('training model')
	logprob=0
	best_model=''
	models=[]
	for i in list(train_arrays.keys()):
		#iterate through the training array
		n_states = 2

		n_observations = len(emission_probs[0])

		start_probs = np.random.dirichlet((1,1),1)[0]

		#can use set transition probabilities or train them with random start-- set is useful to target smaller footprints
		#if you input transition probabilities, only the start probabilities are trained

		transition_probs = np.random.dirichlet((1,1),2)

		model = hmm.MultinomialHMM(n_components=n_states, init_params='', params='st', n_iter=1000)

		model.startprob_ = start_probs
		model.transmat_ = transition_probs
		model.emissionprob_ = emission_probs
		training=train_arrays[i].reshape(-1, 1)
		lengths=len(train_arrays[i])

		print('model', i)
		print('initial starting probabilities:')
		print( model.startprob_)
		print('initial transition probabilities:')
		print( model.transmat_)

		model = model.fit(training, lengths=[lengths])
		#useful to plot output to make sure it's producing the results expected
		#print('plotting results')
		#plot_HMM_results(i, model, array_dic, emission_me_key, emission_pam_key, plot_model, file_prefix)   

		print('final starting probabilities:')
		print( model.startprob_)
		print('final transition probabilities:')
		print( model.transmat_)
		logprob_model=model.monitor_.history[len(model.monitor_.history)-1]
		#print( logprob_model )

		#check best probabilities to choose best model
		if logprob > logprob_model:
			best_num=i
			best_model=model
			logprob=logprob_model   
		models.append(model) 
	print('picked model', best_num)

	return best_model, models

indir, infiles, count, read_count, random_seed, context=options()

print(infiles)

inref=indir+'/Reference/'
context=inref+context

emission_probs=np.array(make_emission_probs(inref+'accessible_probs.tsv', inref+'inaccessible_probs.tsv'))
print(emission_probs.shape)
train_arrays, train_rids=generate_training(indir, infiles, count, read_count, random_seed, context)  
print(train_arrays) 
model, models=trained_HMM(emission_probs, train_arrays)
with open(inref+indir+'_best-model.pickle', 'wb') as handle:
	pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(inref+indir+'_all_models.pickle', 'wb') as handle:
	pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_rids.to_csv(inref+indir+'_training-reads.csv')