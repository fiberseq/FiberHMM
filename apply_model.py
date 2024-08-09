import pandas as pd
import numpy as np
import getopt
import sys
from hmmlearn import hmm
import os
import pickle
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment = None


def options():
    #default parameters
    me_col = 27
    chunk_size = 50000
    train_rids = []
    min_len = 1

    optlist, args = getopt.getopt(sys.argv[1:],'i:g:m:t:o:b:s:l:',
        ['infiles=','context=','model=','train_reads=', 'outdir=', 'me_col=', 'min_len'])

    for o, a in optlist:
        #comma-separated list of paths to input files
        if o=='-i' or o=='--infiles':
            infiles=a.split(',')
        #path to context hdf5
        elif o=='-g' or o=='--context':
            context=a 
        #path to model
        elif o=='-m' or o=='--model':
            model=a     
        #path to tsv of rids used in training
        elif o=='-t' or o=='--train_reads':
            train_rids=pd.read_csv(a)
            train_rids=train_rids['rid'].tolist()
        #path to directory used for output
        elif o=='-o' or o=='--outdir':
            outdir=a  
        #which column are the methylations found in the bedfile. typically 11 or 27 depending on your fibertools output
        elif o=='-b' or o=='--me_col':
            me_col=int(a)  
        #chunk size for chromosome.
        elif o=='-s' or o=='--chunk_size':
            chunk_size=int(a)   
        #minimum footprint count allowed per read
        elif o=='-l' or o=='--min_len':
            min_len=int(a)

    return infiles, context, model, train_rids, outdir, me_col, chunk_size, min_len 


def encode_me(rid, read, read_info, context):

    #grab methylation info, coordinates

    chrom=read_info.loc[rid]['chrom']
    me=np.array(read.dropna())
    me=me[1:-1]
    me=me.astype(int)
    start=read_info.loc[rid,'start']
    end=read_info.loc[rid,'end']

    #generate array with marked methylations

    me=me[np.where(me<(end-start))[0]]
    #mask out the methylations from array for nonmethylated
    no_me=np.arange(end-start)
    no_me=np.delete(no_me, me)

    #read in hexamers for the region, encode the methylations based on the hexamers
    hexamers=pd.read_hdf(context, key=chrom, start=start, stop=end)
    me_encode=hexamers.to_numpy().T[0]
    no_me_encode=me_encode+4097
    #mask each other
    me_encode[no_me]=0
    no_me_encode[me]=0

        #return the full array
    return me_encode+no_me_encode


def unpack(f_pack):
    return np.repeat(f_pack,np.abs(f_pack))

def gaps_to_lengths(fps):
    fpi = np.where(fps > 0)[0]
    fps=np.abs(fps)
    starts = np.cumsum(fps[:fpi[0]])
    lengths = fps[fpi]

    for i in range(1, len(fpi)):
        index = fpi[i]
        starts = np.append(starts, np.sum(fps[:index]))
    
    return starts, lengths

def convert_bed12(reads_df, read_info, chrom, min_len):
    se={}
    le={}
    lle={}

    for index, row in reads_df.iterrows():
        row=row.dropna().to_numpy()
        if len(row)>=min_len:
            starts, lengths = gaps_to_lengths(row)
            lle[index]=str(len(starts))
            starts=','.join(starts.astype(int).astype(str).tolist())
            lengths=','.join(lengths.astype(int).astype(str).tolist())
            se[index]=starts
            le[index]=lengths

    #adding all columns
    read_info=read_info.loc[read_info['chrom']==chrom]
    read_info['blockSizes']=read_info.index.map(le)
    read_info['blockStarts']=read_info.index.map(se)
    read_info['blockCount']=read_info.index.map(lle)
    read_info['thickStart']=read_info['start']
    read_info['thickEnd']=read_info['end']
    read_info['itemRgb']='255,0,0'
    read_info=read_info[['chrom','start','end','rid','thickStart','thickEnd','blockCount','itemRgb','blockSizes','blockStarts']]
    read_info.columns=['chrom','start','end','name','thickStart','thickEnd','blockCount','itemRgb','blockSizes','blockStarts']
    return read_info.dropna()


def apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len):

    print(f"Importing bed", end='\r')
    sys.stdout.flush()

    #read in bedfile
    reads=pd.read_csv(f, usecols=[0,1,2,3,me_col], names=['chrom','start','end', 'rid', 'me'], sep='\t',comment='#')
    reads=reads.loc[reads['chrom'].isin(chromlist)]
    reads=reads.loc[reads['me']!='.']
    read_info=reads.drop(columns=['me'])    
    with tqdm(total=len(chromlist), leave=False) as pbar:

        for chrom in chromlist:
            pbar.set_description(f"Processing {chrom}")

            #filter reads 
            fp_dic={}
            ri = read_info.loc[read_info['chrom']==chrom]
            tmp_out=pd.DataFrame()
            tmp=reads.loc[reads['chrom']==chrom]
            #remove reads used in training if provided
            tmp=tmp.loc[~tmp['rid'].isin(train_rids)]

            #split into chunks to manage memory (related to the .str.split step, this will absolutely nuke your memory if you aren't careful)
            for start in tqdm(range(0, len(tmp), chunk_size),desc="iterating through "+chrom+" chunks", leave=False):
                end = start + chunk_size
                read_chunk = tmp.iloc[start:end]
                read_chunk = read_chunk['me'].str.split(pat=',', expand=True)
                for rid, read in tqdm(read_chunk.iterrows(), total=len(read_chunk), desc="encoding methylations and applying model", leave=False):
                    #encode the read and run it through the HMM
                    read_encode = encode_me(rid, read, ri, context)
                    read_encode = read_encode.astype(int).reshape(-1, 1)
                    pos = model.predict(read_encode)

                    #find the junctions betweens HMM states to id starts and ends of footprints
                    if sum(pos)==0:
                        combined=np.array([-len(pos)])
                    else:
                        #grab footprints and gaps (but won't get last gap)
                        #offset
                        pos_offset=np.append([0],pos)
                        pos=np.append(pos,[0])
                        #difference
                        pos_diff=pos_offset-pos
                        #identify changepoints
                        starts=np.where(pos_diff==-1)[0]
                        ends=np.where(pos_diff==1)[0]
                        ends=np.append([0], ends)
                        #combine into single set of gaps and footprints
                        fps=np.sum((starts*-1,ends[1:]), axis=0).astype('int')
                        gaps=-np.sum((starts,-1*ends[:-1]), axis=0).astype('int')
                        combined = np.vstack((gaps,fps)).reshape((-1,),order='F')
                        
                        #do reverse to grab last gap/footprint
                        #reverse arrays
                        pos_offset=pos_offset[::-1]
                        pos=pos[::-1]
                        #same as before                
                        pos_diff=pos-pos_offset
                        starts=np.where(pos_diff==-1)[0]
                        ends=np.where(pos_diff==1)[0]
                        ends=np.append([0], ends)
                        fps=np.sum((starts*-1,ends[1:]), axis=0).astype('int')
                        gaps=-np.sum((starts,-1*ends[:-1]), axis=0).astype('int')
                        #only care about first (last) gap
                        combined = np.append(combined,[np.vstack((gaps,fps)).reshape((-1,),order='F')[0]])
                        #remove all 0s
                        combined=combined[combined != 0]


                    fp_dic[rid]=combined

            pbar.set_description(f"Writing {chrom}")
            #export the reads as a bed file for easy access
            fp_df=pd.DataFrame.from_dict(fp_dic, orient='index').T
            b12=convert_bed12(fp_df.T, read_info, chrom, min_len)
            dataset=f.split('/')[-1].split('.')[0]
            b12.to_csv(outdir+'/'+dataset+'/'+dataset+'-'+chrom+'_fp.bed', sep='\t', index = False)
            fp_df=''
            pbar.update(1)

infiles, context, model, train_rids, outdir, me_col, chunk_size, min_len = options()

#grab list of chromosomes from context file
tmp=pd.HDFStore(context, 'r')
chromlist=np.array(tmp.keys())
chromlist=np.array([s[1:] for s in chromlist])
tmp.close()

#load model
with open(model, 'rb') as handle:
    model=pickle.load(handle)

#make out directory
if not os.path.exists(outdir):
    os.system('mkdir '+outdir)

#iterate through files
for f in infiles:    
    print('Applying model to', f.rstrip().split('/')[-1])
    dataset=f.split('/')[-1].split('.')[0]
    if not os.path.exists(outdir+'/'+dataset):
        os.system('mkdir '+outdir+'/'+dataset)
    apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len)
    print('Completed', f.rstrip().split('/')[-1])