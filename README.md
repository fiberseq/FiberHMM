How to run the footprint caller:

To simplify input and output, I set the script up to use a set directory structure. 

>HMM_footprint_caller
	>EXPERIMENT_NAME
		>Reference
			reference.fa
			accessible_probs.tsv (if you are not generating them)
			inaccessible_probs.tsv (if you are not generating them)
			chrom_info.tsv

		>Infiles
			>bed
				this is where you put all of your m6a-called bedfiles
				the filename is carried through the whole pipeline, so I would rename them to whateve you want the final file is called
				e.g. untreated_1.bed

		>Outfiles (will be made by the script)

		>Processed (will be made by the script)

The scripts produce quite a few intermediate files

Order of scripts:
1. generate_emission-probs.py
	optional, only if you want to use different control datasets for the emission probabilities

2. prep_bed.py -i EXPERIMENT_NAME -c chrom_info.tsv
	converts the bedfiles to parquet files which are used downstream
	chrom info is just a tab-separated file where the first column is the chromosome name

3. encode_context.py -i EXPERIMENT_NAME -o GENOME.fa
	encodes the genome based on the hexamer sequence context in an hdf5 file for quick lookup

4. train_model.py -i EXPERIMENT_NAME -f SAMPLE1,SAMPLE2,etc. -e GENOME.h5
	optional parameters:
		-c number of iterations to run through
		-r number of total reads to use
		-s random seed for reproducibility
	
	trains the model on a given # of reads from a set of datasets
	outputs the best model as a pickle file into the experiment reference folder, as well as a list of all the models in another pickle

5. apply_model.py -i EXPERIMENT_NAME -f SAMPLE1,SAMPLE2,etc. -e GENOME.h5 -m MODEL.pickle 
	optional parameters:
		-c circular, for now this doesn't work-- I need to fix this
	
	applies the trained model to the rest of the data
	the output is a set of parquet files into the Outfiles folder:
		SAMPLE_read_info.pq -- this is a summary of the chromosomes, rids, starts, and ends of each of the reads to allow for fast lookup
		SAMPLE_chr.pq -- the actual reads in each chromsome. the format is:
			column names:	read#	read#	read#
			each column: accessible, footprint, accessible, footprint, ... accessible, footprint, accessible
		
		the length of footprints and accessible regions are encoded as positive and negative numbers respectively.
