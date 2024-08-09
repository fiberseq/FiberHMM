**Required packages:**
- pandas
- numpy
- os
- getopt
- sys
- hmmlearn (version: 0.2.5)
- pyarrow
- pytables
- pickle
- tqdm

**Script workflow:**
1. **generate_emission-probs.py**  
   Optional, only if you want to use different control datasets for the emission probabilities.  
   I've included accessible and inaccessible probability tsv files which I generated from experimental data as described in our preprint.

3. **encode_context.py -i path_to/genome_name.fa -o path_to/genome_name.h5**  
   Encodes the genome based on the hexamer sequence context in an hdf5 file for quick lookup. Only needs to be run once per genome.

4. **train_model.py -i path_to/dataset_1.bed,path_to/dataset_2.bed,etc. -g path_to/genome_name.h5 -p path_to/accessible_probs.tsv,path_to/inaccessible_probs.tsv -o path_to_output_directory**  
   Optional parameters:
   - `-c` number of iterations to run through
   - `-r` number of total reads to use across all datasets
   - `-s` random seed for reproducibility
   - `-b` column # (0-based) in bed files with methylation starts. 11 if just the m6A output from fibertools, 27 (default) if full output
   - `-o` a directory path where the output files will be stored (models, list of reads used in training).

   Trains the model on a given number of reads from a set of datasets.  
   Outputs the best model and a list of all the models in pickles and a tsv of reads used in the training. Only needs to be run once per genome-- the model parameters shouldn't vary very much between similar datasets.
   
6. **apply_model.py -i path_to/dataset_1.bed,path_to/dataset_2.bed,etc. -m path_to/best_model.pickle -t path_to/training-reads.tsv -g path_to/genome_name.h5 -p path_to/accessible_probs.tsv,path_to/inaccessible_probs.tsv -o path_to_output_directory**
   <br>
   Optional parameters:
   - `-l` minimum footprints allowed per read, default = 1
   
   Applies the trained model to the rest of the data. The output is in the bed12 format, with footprint starts and lengths stored. Note that any footprints overlapping the start or end of the read are of unclear length and this should be taken into account for downstream analyses.
