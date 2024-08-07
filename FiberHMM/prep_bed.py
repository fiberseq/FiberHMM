import pandas as pd
import numpy as np
import os
import getopt
import sys
from matplotlib import pyplot as plt

#this converts bed files to split parquet files which minimized memory impact of the downstream steps
#only takes an input of the directory containing the bed files

def options():

	chromlist=['chr2R', 'chr2L','chr3R','chr3L', 'chrX','chrY','chr4']
	
	optlist, args = getopt.getopt(sys.argv[1:],'i:e:c:m:s:',
		['infile=', 'chrom_info=', 'encoding=', 'min-me=','chunk_size='])
	for o, a in optlist:
		if o=='-i' or o=='--infile':
			indir=a
	for o, a in optlist:
		if o=='-c' or o=='--chrom_info':
			chrom_info=indir+'/Reference/'+a
			chromlist=[]
			for line in open(chrom_info):
				line=line.rstrip().split('\t')
				chromlist.append(line[0])
		
	return indir, chromlist


def bed_to_csv(infile, outdir, prefix, chromlist):
	#makes a directory
	os.system('mkdir '+outdir)

	#reads in the reads as a dataframe, columns could be wrong down the line if the m6a caller changes or something
	reads=pd.read_csv(infile, usecols=[0,1,2,3,11], names=['chrom','start','end', 'rid', 'me'], sep='\t')
	reads=reads.loc[reads['me']!='.']
	reads=reads.sort_values(by=['chrom','start'])
	reads=reads.reset_index(drop=True)
	reads.index=reads.index.astype(str)

	#only grab the chromosomes we actually want, I'll add an option to input a chrominfo file 
	for chrom in chromlist:
		tmp=reads.loc[reads['chrom']==chrom]
		tmp=tmp['me'].str.split(pat=',', expand=True)
		tmp.to_csv(outdir+'/'+prefix+'_'+chrom+'.csv')

	#write out to csv
	reads=reads.drop(columns=['me'])
	reads=reads.loc[reads['chrom'].isin(chromlist)]
	reads.to_csv(outdir+'/'+prefix+'_read-info.csv')

def bed_to_parquet(infile, outdir, prefix, chromlist):
	#makes a directory
	os.system('mkdir '+outdir)

	#reads in the reads as a dataframe, columns could be wrong down the line if the m6a caller changes or something
	reads=pd.read_csv(infile, usecols=[0,1,2,3,11], names=['chrom','start','end', 'rid', 'me'], sep='\t')
	reads=reads.loc[reads['me']!='.']
	reads=reads.sort_values(by=['chrom','start'])
	reads=reads.reset_index(drop=True)
	reads.index=reads.index.astype(str)

	#only grab the chromosomes we actually want, I'll add an option to input a chrominfo file 
	for chrom in chromlist:
		tmp=reads.loc[reads['chrom']==chrom]
		tmp=tmp['me'].str.split(pat=',', expand=True).T
		tmp.to_parquet(outdir+'/'+prefix+'_'+chrom+'.pq')

	#write out to parquet
	reads=reads.drop(columns=['me'])
	reads=reads.loc[reads['chrom'].isin(chromlist)]
	reads.to_parquet(outdir+'/'+prefix+'_read-info.pq')



indir, chromlist =options()

indir=indir+'/Infiles/bed'

#iterates through all bed files

os.system('mkdir '+indir.replace('bed','parquet'))
os.system('mkdir '+indir.replace('bed','context'))

for f in os.listdir(indir):
	print(f)
	infile=indir+'/'+f
	prefix=f.replace('.bed','')
	outdir1=indir.replace('bed','parquet')+'/'+prefix
	print('converting to split methylation parquet file')
	bed_to_parquet(infile,outdir1,prefix, chromlist)
