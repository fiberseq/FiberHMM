How to run the footprint caller:

Required packages:
pandas (version: 2.0.0)
numpy
os
getopt
sys
hmmlearn (version:0.2.5)
pyarrow (version: 11.0.0)
pytables (3.8.0)
pickle (0.7.5)


To simplify input and output, I set the script up to use a set directory structure. 

>HMM_footprint_caller
	>EXPERIMENT_NAME
		>Reference
			GENOME.fa
			accessible_probs.tsv (if you are not generating them)
			inaccessible_probs.tsv (if you are not generating them)
			chrom_info.tsv
			(I've included an example for each of these from drosophila, but you will need to make your own GENOME.fa and chrom_info.tsv for the plasmid)

		>Infiles
			>bed
				this is where you put all of your m6a-called bedfiles
				the filename is carried through the whole pipeline, so I would rename them to whatever you want the final file to be called
				e.g. untreated_1.bed

		>Outfiles (will be made by the script)

		>Processed (will be made by the script)

The scripts produce quite a few intermediate files that help cut down on memory usage and give you checkpoints in the pipeline. 
The final output file is very small, but the intermediate files are pretty big so make sure you have space
I don't have a single-line unified pipeline yet, rather it's a set of scripts that you can run manually in order.

Order of scripts:
1. generate_emission-probs.py
	optional, only if you want to use different control datasets for the emission probabilities. 
	I've included accessible and inaccessible probability files which I generated from the latest m6A caller results.

2. prep_bed.py -i EXPERIMENT_NAME -c chrom_info.tsv
	When I use ft extract --m6a, my chromosome name is the zmw_id. I have to manually change this to the chromosome name using the following command
	'sed 's/^[^\t]*/linear_LeaGFP/' your_file.bed12 > modified_file.bed12'
	converts the bedfiles to split parquet files which are used downstream
	chrom info is just a tab-separated file where the first column is the chromosome name

3. encode_context.py -i EXPERIMENT_NAME -f path_to_GENOME.fa
	encodes the genome based on the hexamer sequence context in an hdf5 file for quick lookup

4. train_model.py -i EXPERIMENT_NAME -f SAMPLE1,SAMPLE2,etc. -e GENOME.h5
	optional parameters:
		-c number of iterations to run through
		-r number of total reads to use
		-s random seed for reproducibility
	
	trains the model on a given # of reads from a set of datasets
	outputs the best model as a pickle file into the experiment reference folder, as well as a list of all the models in another pickle
	also outputs a list of reads used in the training as a csv file in the reference folder in case you want to avoid using those reads downstream

5. apply_model.py -i EXPERIMENT_NAME -f SAMPLE1,SAMPLE2,etc. -e GENOME.h5 -m MODEL.pickle 
	optional parameters:
		-c circular, for now this doesn't work-- I need to fix this
	
	applies the trained model to the rest of the data
	the output is a set of parquet files in subfolders in the Outfiles folder:
		SAMPLE_read_info.pq -- this is a summary of the chromosomes, rids, starts, and ends of each of the reads to allow for fast lookup
		SAMPLE_chr.pq -- the actual reads in each chromsome. the format is:
			column names:	read#	read#	read#
			each column: accessible, footprint, accessible, footprint, ... accessible, footprint, accessible
		
		the length of footprints and accessible regions are encoded as positive and negative numbers respectively. I will send you a jupyter notebook running through how I decode and analyze this file
