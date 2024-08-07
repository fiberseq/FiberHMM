import pandas as pd
import numpy as np
import getopt
import sys

def options():
	optlist, args = getopt.getopt(sys.argv[1:],'i:f:',
		['infile=','fasta='])

	for o, a in optlist:
		if o=='-i' or o=='--infile':
			indir=a
		elif o=='-f' or o=='--fasta':
			fasta=a
		
	return indir, fasta

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
	print('importing fasta')
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
		print(len(fdic[chrom]))
	return fdic

def hexamer_context():
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

def encode_context(fa_dic, hexamers_all, outfile):

    context_dic={}
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
        context_df=pd.DataFrame(context_dic)
        context_dic={}
        
        context_df=context_df.astype(float)
        context_df.to_hdf(outfile, chrom, format='table')
        context_df=''

indir, fasta = options()
ref=indir+'/Reference/'

fa_dic=make_fa_dic(ref+fasta)
hexamers=hexamer_context()
encode_context(fa_dic, hexamers, ref+fasta.replace('.fa','.h5'))