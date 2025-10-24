import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import getopt
import sys
from tqdm import tqdm
import os


def options():
    optlist, args = getopt.getopt(sys.argv[1:],'i:o:',
        ['infile=','outfile='])

    for o, a in optlist:
        if o == '-i' or o == '--infile':
            infile = a
        elif o == '-o' or o == '--outfile':
            outfile = a
        
    return infile, outfile


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
    with tqdm(open(infile, 'r'), desc="counting lines in fasta") as f:
        total_lines = sum(1 for line in f)
    with open(infile, 'r') as f:
        for line in tqdm(f, desc="Importing fasta", leave=False, total = total_lines):
            line=line.rstrip()
            if line.startswith('>'):
                chrom=line.split(' ')[0].replace('>','') #grab chromosome name up until first whitespace
                if '-' in chrom: #replace forbidden characters
                    chrom=chrom.replace('-','__')
                if ':' in chrom:
                    chrom=chrom.replace(':','___')
                if '.' in chrom:
                   chrom=chrom.replace(':','____') 

                chrom_filter=True   # This is preserved in case I want to hardcode leaving out specific chromosomes.
                                    # This can be useful in weird assemblies with many 1000s of contigs if you don't
                                    # actually need all of them. If you want to use it, add an "if" statement that
                                    # accounts for the string you want to filter out and set chrom_filter to False.
                                    # If this script is slow or produces a huge database but you just need one or two
                                    # chromosomes/contigs, consider using this option.
                fdic[chrom]=[]

            elif chrom_filter:
                fdic[chrom].append(line.upper())
    for chrom in fdic:
        fdic[chrom]=''.join(fdic[chrom])
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
    
    with tqdm(total=len(fa_dic), desc="Encoding chromosome") as pbar:
        for chrom in fa_dic:
            pbar.set_description(f"Encoding {chrom}")

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

            pbar.update(1)

infile, outfile = options()

#check if context already exists and if so delete it
if os.path.exists(outfile):
    os.remove(outfile)

fa_dic = make_fa_dic(infile)
hexamers = hexamer_context()
encode_context(fa_dic, hexamers, outfile)
