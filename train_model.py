import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import getopt
import sys
from hmmlearn import hmm
import os
import pickle
from tqdm import tqdm
import h5py
tqdm.pandas()
pd.options.mode.chained_assignment = None


def options():
    #default parameters
    count = 10
    read_count = 150
    random_seed = 1
    me_col = 27
    edge_trim = 10


    optlist, args = getopt.getopt(sys.argv[1:],'i:c:r:s:g:b:p:o:e:',
        ['infiles=','count=','read_count=','random_seed=', 'context=','me_col=', 'probs=', 'outdir=', 'edge_trim='])

    for o, a in optlist:
        if o=='-i' or o=='--infiles':
            infiles=a.split(',')
        elif o=='-c' or o=='--count':
            count=int(a)
        elif o=='-r' or o=='--read_count':
            read_count=int(a)
        elif o=='-s' or o=='--random_seed':
            random_seed=int(a)  
        elif o=='-g' or o=='--context':
            context=a   
        elif o=='-b' or o=='--me_col':
            me_col=int(a)   
        elif o=='-p' or o=='--probs':
            acc_in, inacc_in=a.split(',')
        elif o=='-o' or o=='--outdir':
            outdir = a
        elif o == '-e' or o == '--edge_trim':
            edge_trim = int(a)
    return infiles, count, read_count, random_seed, context, me_col, acc_in, inacc_in, outdir, edge_trim

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


def encode_me(rid, read, read_info, context, edge_trim):
    #grab info, read
    chrom = read_info.loc[rid]['chrom']
    start = read_info.loc[rid, 'start']
    end = read_info.loc[rid, 'end']

    me = np.array(read.dropna())[1:-1]
    #remove any methylations in the trim region
    me = me.astype(int)
    me = me[np.where(
        (me>edge_trim)&(me<((end-start)-edge_trim))
        )]

    #make sure within range, find positions with no methylation
    me = me[np.where(me < (end - start))[0]]
    no_me = np.arange(end - start)
    no_me = np.delete(no_me, me)

    #grab sequence context info from context file 
    with h5py.File(context, 'r', swmr=True) as f:
        hexamers = f[chrom]['table'][(start+edge_trim):(end-edge_trim)]

    #encode methylations and no methylations
    me_encode = np.array([item[1] for item in hexamers]).T[0]
    #add non-A (so, 0% probability of methylation) to edge bases
    me_encode = np.pad(me_encode, pad_width=(edge_trim, edge_trim), mode='constant', constant_values=4096)
    no_me_encode = me_encode + 4097

    #zero out me/no me positions
    me_encode[no_me] = 0
    no_me_encode[me] = 0

    #add, return
    return me_encode + no_me_encode


def import_bed(f, me_col):
    s_adjust={}
    e_adjust={}
    drops=[]    
    reads=pd.read_csv(f, usecols=[0,1,2,3,me_col], names=['chrom','start','end', 'rid', 'me'], sep='\t',comment='#')
    reads=reads.loc[reads['chrom'].isin(chromlist)]
    reads=reads.loc[reads['me']!='.']
    return reads

def generate_training(infiles, count, read_count, random_seed, context, me_col, edge_trim):

    train_df=pd.DataFrame()
    read_count=read_count//len(infiles)

    train_rids=[]

    with tqdm(total=len(infiles)) as pbar:
        for f in infiles:
            #import data
            pbar.set_description(f"Importing {f.rstrip().split('/')[-1]}")
            reads = import_bed(f, me_col)
            #sample data
            pbar.set_description(f"Encoding {f.rstrip().split('/')[-1]}")
            train_reads = reads.sample(n=read_count, random_state=random_seed)
            ri=train_reads.drop(columns=['me'])
            train_reads = train_reads['me'].str.split(pat=',', expand=True)
            
            #saving list of reads used in training
            train_rids+=ri['rid'].to_list()
        
            train_dic={}    
            for rid, read in train_reads.iterrows():
                #encode the read and run it through the HMM
                read_encode = encode_me(rid, read, ri, context, edge_trim)
                train_dic[rid]=read_encode

            tmp=pd.DataFrame.from_dict(train_dic, orient='index')
            train_df=pd.concat([train_df,tmp])
            pbar.update(1)
        pbar.set_description("Completed sampling")


    train_arrays={}
    for i in range(count):
        train_df=train_df.sample(frac=1, random_state=i)
        train_array=train_df.to_numpy().flatten()
        train_array = train_array[~np.isnan(train_array)].astype(int)
        train_arrays[i]=train_array

    return train_arrays, train_rids

def train_HMM(emission_probs, train_arrays):
    #train HMM using dataset
    logprob=0
    best_model=''
    models=[]

    with tqdm(total=len(list(train_arrays.keys()))) as pbar:
        j=0
        for i in list(train_arrays.keys()):
            #iterate through the training array
            pbar.set_description(f"Training model {j+1}")
            
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

            model = model.fit(training, lengths=[lengths])
            
            logprob_model=model.monitor_.history[len(model.monitor_.history)-1]

            #check best probabilities to choose best model
            if logprob > logprob_model:
                best_num=i
                best_model=model
                logprob=logprob_model   

            models.append(model) 
            j+=1
            pbar.update(1)
        pbar.set_description("Completed training")

    print('Picked model', best_num)
    print('Final starting probabilities:')
    print( model.startprob_)
    print('Final transition probabilities:')
    print( model.transmat_)

    return best_model, models

infiles, count, read_count, random_seed, context, me_col, acc_in, inacc_in, outdir, edge_trim =options()

if not os.path.exists(outdir):
    os.makedirs(outdir)

tmp=pd.HDFStore(context, 'r')
chromlist=np.array(tmp.keys())
chromlist=np.array([s[1:] for s in chromlist])
tmp.close()

emission_probs=np.array(make_emission_probs(acc_in, inacc_in))
train_arrays, train_rids=generate_training(infiles, count, read_count, random_seed, context, me_col, edge_trim)  
model, models=train_HMM(emission_probs, train_arrays)
with open(outdir+'/best-model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(outdir+'/all_models.pickle', 'wb') as handle:
    pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

pd.DataFrame(train_rids, columns=['rid']).to_csv(outdir+'/training-reads.tsv')