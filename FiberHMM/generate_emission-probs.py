import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import getopt
import sys


#required options:
#	-i : path to input file
#	-o : path to output file
#	-f : path to reference fasta sequence
#	-c : number of reads to load in at once from bed file (useful to manage memory)
#	-t : total reads to run, no need to run a bunch

def options():
	#default parameters
	min_me_frac=0
	min_read_len=1000
	chunksize=1000000
	intermediate_save=10000
	total_reads=100000

	optlist, args = getopt.getopt(sys.argv[1:],'i:o:f:c:m:r:s:t:',
		['infile=','outfile=','fasta=', 'chunksize=','min_read_len=','min_me=','intermediate_save=', 'total_reads='])

	for o, a in optlist:
		if o=='-i' or o=='--infile':
			infile=a
		elif o=='-o' or o=='--outfile':
			outfile=a
		elif o=='-f' or o=='--fasta':
			fasta=a
		elif o=='-c' or o=='--chunksize':
			chunksize=a
		elif o=='-m' or o=='--min_me':
			min_me_frac=float(a)
		elif o=='-r' or o=='--min_read_len':
			min_read_len=int(a)
		elif o=='-s' or o=='--intermediate_save':
			intermediate_save=int(a)
		elif o=='-t' or o=='--total_reads':
			total_reads=int(a)		
	return infile, outfile, fasta, chunksize, min_me_frac, min_read_len, intermediate_save,total_reads

def rc(seq):
	#find the reverse complement of a sequence
	rc_seq=''
	seq=list(seq)
	seq.reverse()
	rdic={'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
	for base in seq:
		rc_seq+=rdic[base]
	return rc_seq

def make_fa_dic(infile):
	#import dictionary of sequences from fasta file
	#sequence is a string, each entry is chromosome
	#for fast lookup of sequence context
	fdic={}
	chrom_filter=False
	for line in open(infile):
		line=line.rstrip()
		if '>' in line and 'sequence' not in line and 'genome' not in line:
			if '>chr' in line:
				chrom=line.replace('>','')
			else:
				line=line.split(' ')
				chrom='chr'+line[len(line)-1]
			chrom_filter=True
			fdic[chrom]=[]
		elif '>' in line and 'sequence' in line or 'genome' in line:
			chrom_filter=False
		elif chrom_filter:
			fdic[chrom].append(line.upper())
	for chrom in fdic:
		fdic[chrom]=''.join(fdic[chrom])
	return fdic

def hexamer_context():
	#generate a dictionary encoding all possible hexamer contexts around an A with a numerical code
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
	hexamers_all=hexamers.copy()
	for hexamer in hexamers:
		hexamers_all[rc(hexamer)]=hexamers[hexamer]
	return hexamers_all

def encode_context(fa_dic, hexamers_all):
	#assign code to each position in genome
	context_dic={}
	context_df=pd.DataFrame()
	cd=False
	
	for chrom in fa_dic:
		print(chrom)
		tmp=[4096]*3
		for pos in range(3,len(fa_dic[chrom])-3):
			hexamer=fa_dic[chrom][pos-3:pos+3+1]
			if hexamer in hexamers_all:
				tmp.append(hexamers_all[hexamer])
			else:
				tmp.append(4096)
		for i in range(3):
			tmp.append(4096)
			
		context_dic[chrom]=dict(zip(range(1,len(tmp)+1), tmp))
		cd=pd.DataFrame(context_dic)
		context_dic={}
		
		cd=cd.astype(float)
		context_df=pd.concat([context_df, cd], axis=1)
	return context_df

#options
infile,outfile,fasta,chunksize,min_me_frac,min_read_len,intermediate_save, total_reads=options()

#generate hexamer context codes
hex_to_code = hexamer_context()
#import fasta reference
print('importing fasta')
fa_dic=make_fa_dic(fasta)
#generate context code dataframe for every position in genome
print('generating context reference')
context_df=encode_context(fa_dic, hex_to_code)
fasta=''

print('reading in data')
j=0
for reads in pd.read_csv(infile, usecols=[0,1,2,3,11], names=['chrom','start','end', 'rid', 'me'], sep='\t', chunksize=int(chunksize)):
	#filter reads by length/methylation status/chromosome annotation
	reads=reads.loc[(reads['end']-reads['start'])>=int(min_read_len)]
	reads=reads.loc[reads['me']!='.']
	reads=reads.loc[reads['chrom'].isin(context_df.columns)]

	#split read methylations, filter by methylation fraction
	tmp=reads['me'].str.split(pat=',', expand=True)
	tmp['me_ct']=tmp.count(axis=1)
	tmp['start']=reads['start']
	tmp['end']=reads['end']
	tmp['chrom']=reads['chrom']
	tmp['me_frac']=tmp['me_ct']/(tmp['end']-tmp['start'])
	tmp=tmp.loc[tmp['me_frac']>float(min_me_frac)]
	tmp=tmp.drop(['me_ct','me_frac'], axis=1)
	reads=''

	print('analyzing reads')
	start=time.time()
	for i in tmp.index:
		#find methyl/no methyl positions
		row=tmp.loc[i,:].copy()
		hits=row.drop(['start','end','chrom'])
		hits=hits.dropna().astype(int).to_numpy()[:-1]
		hits=hits[hits>0]
		no_hits=np.zeros(int(row['end'])-int(row['start']))
		no_hits[hits]+=1
		no_hits=np.where(no_hits<1)[0][1:]
		hits=hits+int(row['start'])+1
		no_hits=no_hits+int(row['start'])+1

		#find the context codes for each in the context dataframe
		h=context_df.loc[hits,row['chrom']]
		nh=context_df.loc[no_hits,row['chrom']]
		
		#count the context codes 
		h=pd.DataFrame([np.zeros(h.shape[0])+1,h.tolist()]).T
		h.columns=h.columns.astype(str)
		h=h.groupby('1').sum()
		nh=pd.DataFrame([np.zeros(nh.shape[0])+1,nh.tolist()]).T
		nh.columns=nh.columns.astype(str)
		nh=nh.groupby('1').sum()

		#merge with the total counts
		if j==0:	
			h_a=h
			nh_a=nh
		else:
			h_a=pd.concat([h,h_a])
			h_a=h_a.groupby('1').sum()
			nh_a=pd.concat([nh,nh_a])
			nh_a=nh_a.groupby('1').sum()

		j+=1
		#print(j)
		if j%int(intermediate_save)==0:
			print('analyzed '+str(j)+' reads in '+str(time.time()-start)+' seconds')

			#write to tsv output
			print('writing intermediate output')
			a=pd.concat([h_a, nh_a], axis=1)
			a.columns=['hit', 'nohit']
			a=a.fillna(0)
			a['ratio']=a['hit']/(a['hit']+a['nohit'])
			a=a.sort_index()
			a.to_csv(outfile, sep='\t')
		if j==total_reads:
			print('completed requested analysis of '+str(total_reads)+' reads')
			break
	if j==total_reads:
		break
#write to tsv output
print('writing final output')
a=pd.concat([h_a, nh_a], axis=1)
a.columns=['hit', 'nohit']
a=a.fillna(0)
a['ratio']=a['hit']/(a['hit']+a['nohit'])
a=a.sort_index()
a.to_csv(outfile, sep='\t')