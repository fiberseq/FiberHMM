import pandas as pd
import numpy as np
import getopt
import sys
from hmmlearn import hmm
import os
import pickle

def options():
    #default parameters

    optlist, args = getopt.getopt(sys.argv[1:],'i:f:e:m:',
        ['indir=','infiles=','context=','model='])

    for o, a in optlist:
        if o=='-i' or o=='--indir':
            indir=a
        elif o=='-f' or o=='--infiles':
            infiles=a.split(',')
        elif o=='-e' or o=='--context':
            context=a 
        elif o=='-m' or o=='--model':
            model=a     
    return indir, infiles, context, model


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
    return np.repeat(f_pack, np.abs(f_pack))

def apply_model(model, f, indir, outdir, context):

    read_info=pd.read_parquet(indir+f+'/'+f+'_read-info.pq')
    s_adjust={}
    e_adjust={}
    drops=[]

    print('applying model')
    
    for chrom in read_info['chrom'].unique():
        print(chrom)
        fp_dic={}
        ri = read_info.loc[read_info['chrom']==chrom]
        reads = pd.read_parquet(indir+f+'/'+f+'_'+chrom+'.pq').T
        i=0
        for rid, read in reads.iterrows():
            #encode the read and run it through the HMM
            read_encode = encode_me(rid, read, ri, context)
            read_encode = read_encode.astype(int).reshape(-1, 1)
            pos = model.predict(read_encode)

            #find the junctions betweens HMM states to id starts and ends of footprints
            pos_offset=np.append([0],pos)
            pos=np.append(pos,[0])
            pos_diff=pos_offset-pos
            starts=np.where(pos_diff==-1)[0]
            ends=np.where(pos_diff==1)[0]
            ends=np.append([0], ends)
            
            #identify lengths of footprints and gaps based on starts/ends
            #save them as +length, -length respectively
            fps=np.sum((starts*-1,ends[1:]), axis=0).astype('int')
            gaps=-np.sum((starts,-1*ends[:-1]), axis=0).astype('int')
            combined = np.vstack((gaps,fps)).reshape((-1,),order='F')
            print(combined)
            print(unpack(combined))
          #  print(combined)
            #need to throw out the first and last footprints bc they are cut off and the wrong size
            #also need to adjust the start and end of the read to account for the missing footprints
            new_s=ri.loc[rid]['start']+combined[1]
            new_e=ri.loc[rid]['end']-combined[-1]
            combined=combined[2:-1]
            s_adjust[rid]=new_s
            e_adjust[rid]=new_e
            fp_dic[rid]=combined
            i+=1
            if i%10000==0:
                print('completed '+str(i)+' reads')

        #export the reads as a parquet file for easy access
        fp_df=pd.DataFrame.from_dict(fp_dic, orient='index').T
        fp_df.to_parquet(outdir+f+'/'+f+'_'+chrom+'_footprints.pq')
        fp_df=''

    #generate new read-info file to account for changes to start/end
    #print(read_info['start'])
    read_info['start']=read_info.index.map(s_adjust)
    #print(read_info['start'])
    read_info['end']=read_info.index.map(e_adjust)
    read_info.to_parquet(outdir+f+'/'+f+'_read-info.pq') 


indir, infiles, context, model = options()

inpq=indir+'/Infiles/parquet/'
inref=indir+'/Reference/'
context=inref+context
outdir=indir+'/Outfiles/'


with open(inref+model, 'rb') as handle:
    model=pickle.load(handle)
for f in infiles:
    print(f)
    os.system('mkdir '+outdir+f)
    apply_model(model, f, inpq, outdir, context)

