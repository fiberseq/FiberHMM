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
import tempfile
import glob
import h5py
import time



def options():
    #default parameters
    me_col = 28
    chunk_size = 50000
    train_rids = []
    min_len = 0
    circle=False
    edge_trim = 10

    optlist, args = getopt.getopt(sys.argv[1:],'i:g:m:t:o:b:s:l:re:',
        ['infile=','context=','model=','train_reads=', 'outdir=', 'me_col=', 'min_len=','circular', 'edge_trim='])

    for o, a in optlist:
        #path to input file
        if o=='-i' or o=='--infile':
            infile=a
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
        #turn on circular mode for full circular genome reads (tiles the encoded read 3x, allowing for the edges of the read to be properly footprint called). 
        #the output is a nonstandard bed12, which has 3x the footprints
        elif o == '-r' or o == '--circular':
            circle = True
        elif o == '-e' or o == '--edge_trim':
            edge_trim = int(a)

    return infile, context, model, train_rids, outdir, me_col, chunk_size, min_len, circle, edge_trim


def encode_me(rid, read, read_info, context, circle, edge_trim):
    #grab info, read
    chrom = read_info.loc[rid]['chrom']
    start = read_info.loc[rid, 'start']
    end = read_info.loc[rid, 'end']

    me = np.array(read.dropna())[1:-1]
    #remove any methylations in the trim region
    me = me.astype(int)
    me = me[np.where(
        (me>(edge_trim+start))&(me<(end-edge_trim))
        )]

    #make sure within range, find positions with no methylation
    me = me[np.where(me < (end))[0]]-start
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
    if not circle:
        return me_encode + no_me_encode
    else:
        me_encode = me_encode + no_me_encode
        return np.tile(me_encode, 3)

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

def apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, circle, edge_trim):

    dataset=f.split('/')[-1].split('.')[0]
    sys.stdout.flush()

    #read in bedfile, grab reads from valid chromosomes
    chrom=''
    reader = pd.read_csv(f, usecols=[0, 1, 2, 3, me_col], names=['chrom', 'start', 'end', 'rid', 'me'], sep='\t', comment='#', chunksize=chunk_size)
    
    with tqdm(total=len(chromlist), leave=True) as pbar:
        i=0
        for chunk in reader:
            i+=1
            #make sure chromosomes are in the encoding
            chunk = chunk.loc[chunk['chrom'].isin(chromlist)]
            
            #update displayed chromosome based on the first chromosome in the chunk
            chrom_new=chunk['chrom'].tolist()[0]
            if chrom!=chrom_new:  
                chrom=chrom_new
                pbar.set_description(f"Processing {chrom}")
                pbar.update(1)

            #remove reads used in training if provided
            chunk=chunk.loc[~chunk['rid'].isin(train_rids)]
            
            #generate bed12 for reads with no methylation (so, all one footprint)
            no_me_b12 = chunk.loc[chunk['me'] == '.'].drop('me', axis=1)
            no_me_b12['thickStart'] = no_me_b12['start']
            no_me_b12['thickEnd'] = no_me_b12['end']
            no_me_b12['itemRgb'] = '255,0,0'
            no_me_b12['blockCount'] = 1
            no_me_b12['blockStarts'] = 1
            no_me_b12['blockSizes'] = no_me_b12['end'] - no_me_b12['start']

            if not circle:
                no_me_b12['blockSizes'] = no_me_b12['end'] - no_me_b12['start']
            else:
                no_me_b12['blockSizes'] = no_me_b12['end']*3 - no_me_b12['start']

            no_me_b12.columns = ['chrom', 'start', 'end', 'name', 'thickStart', 'thickEnd', 'blockCount', 'itemRgb', 'blockSizes', 'blockStarts']
            chrom = chunk['chrom'].iloc[0]

            #generate bed12 for reads with methylation
            chunk=chunk.loc[chunk['me']!='.']
            b12 = chunk.drop(columns=['me'])
            b12['thickStart'] = b12['start']
            b12['thickEnd'] = b12['end']
            b12['itemRgb'] = '255,0,0'

            #grab methylations
            chunk = chunk['me'].str.split(pat=',', expand=True)

            all_starts=[]
            all_lengths=[]
            all_counts=[]

            #encode methylations, predict footprints
            with tqdm(total=chunk_size, initial=no_me_b12.shape[0], leave=False) as pbar2:
                pbar2.set_description(f"Applying model to chunk {i}")

                for index, read in chunk.iterrows():
                    read_encode = encode_me(index, read, b12, context, circle, edge_trim)
                    read_encode = read_encode.astype(int).reshape(-1, 1)
                    pos = model.predict(read_encode)

                    #if no footprints, store '.'
                    if sum(pos) == 0:
                        all_starts.append('.')
                        all_lengths.append('.')
                        all_counts.append(0)

                    #else calculate footprint starts and lengths from probabilities
                    else:
                        pos_offset = np.append([0], pos)
                        pos = np.append(pos, [0])
                        pos_diff = pos_offset - pos
                        starts = np.where(pos_diff == -1)[0]
                        ends = np.where(pos_diff == 1)[0]
                        ends = np.append([0], ends)
                        lengths = np.sum((starts * -1, ends[1:]), axis=0).astype('int')
                        all_starts.append(','.join(starts.astype(str)))
                        all_lengths.append(','.join(lengths.astype(str)))
                        all_counts.append(len(starts))

                    pbar2.update(1)

                pbar2.set_description(f"Writing chunk {i}")

                #add to bed12
                b12['blockCount'] = all_counts
                b12['blockStarts'] = all_starts
                b12['blockSizes'] = all_lengths

                #combine, sort bed12s
                b12 = b12.rename(columns={'rid': 'name'})
                b12.columns = ['chrom', 'start', 'end', 'name', 'thickStart', 'thickEnd', 'blockCount', 'itemRgb', 'blockStarts', 'blockSizes']
                b12 = pd.concat([b12, no_me_b12])
                b12 = b12.sort_values(by=['chrom', 'start'])

                # Write to a temporary file (split by chromosome if necessary)
                tmp_file = os.path.join(tmp_dir, f"{dataset}_{i}.bed")
                b12.to_csv(tmp_file, sep='\t', index=False)



f, context, model, train_rids, outdir, me_col, chunk_size, min_len, circle, edge_trim = options()

#grab list of chromosomes from context file
tmp=pd.HDFStore(context, 'r')
chromlist=np.array(tmp.keys())
chromlist=np.array([s[1:] for s in chromlist])
chromlist = chromlist[~np.char.find(chromlist, '_') >= 0]
tmp.close()

#make out directory and temporary directory
if not os.path.exists(outdir):
    os.system('mkdir '+outdir)
tmp_dir = tempfile.mkdtemp(dir=outdir)  # Create a temporary directory for storing temporary files

#load model
with open(model, 'rb') as handle:
    model=pickle.load(handle)

#grab dataset name
dataset=f.split('/')[-1].split('.')[0]

#run model
apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, circle, edge_trim)

#need to pause for 1s for small datasets
time.sleep(1)

#combine temp files, remove
print('Combining temporary files')
tmp_files = glob.glob(os.path.join(tmp_dir, f"{dataset}_*.bed"))
final_file = os.path.join(outdir, f"{dataset}_fp.bed")
with open(final_file, 'w') as fout:
    for tmp_file in tmp_files:
        with open(tmp_file, 'r') as fin:
            fout.write(fin.read())
        os.remove(tmp_file) 

os.rmdir(tmp_dir)