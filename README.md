## **Required Packages:**

- **pandas**
- **numpy**
- **os**
- **getopt**
- **sys**
- **hmmlearn** (version: 0.2.5)
- **pyarrow**
- **pytables**
- **pickle**
- **tqdm**
- **h5py**
- **temnpfile**
- **multiprocessing** (for multiprocess version)
- **threading** (for multiprocess version)
- **logging** (for multiprocess version)

---

## **Script Workflow:**

### 1. **Generate emission probabilities (optional)**
   - **Command:** `generate_emission-probs.py -i path_to/control_dataset -o path_to/output.tsv -f path_to/genome_name.fa`
   - - **Optional Parameters:**
     - `-c` Number of reads to load at once (default: 1000000).
     - `-t` Total number of reads to run (default: 100000).
     - `-m` Minimum methylation fraction (default: 0).
     - `-r` Minimum read length (default: 1000).
     - `-s` Count of reads until you write a checkpoint output (default: 10000).
   - **Purpose:** Generates emission probabilities if you want to use your own different control datasets.
   - **Note:** Pre-generated accessible and inaccessible probability TSV files are provided, based on experimental data described in our manuscript. Even with different organisms, changes to PacBio technology, and improvements to methylation calling we haven't found major changes in these control datasets. It's also not really necessary to run this on an entire dataset: the default parameters for total read count is more than enough to get stable probabilities. You will need to run this on both an accessible and inaccessible dataset (see our manuscript).

### 2. **Encode context**
   - **Command:** `encode_context.py -i path_to/genome_name.fa -o path_to/genome_name.h5`
   - **Purpose:** Encodes the genome based on the hexamer sequence context into an HDF5 file for quick lookup.
   - **Usage:** Run once per genome.

### 3. **Train model**
   - **Command:** `train_model.py -i path_to/dataset_1.bed,path_to/dataset_2.bed,etc. -g path_to/genome_name.h5 -p path_to/accessible_probs.tsv,path_to/inaccessible_probs.tsv -o path_to_output_directory`
   - **Optional Parameters:**
     - `-c` Number of iterations to run.
     - `-r` Total number of reads to use across all datasets.
     - `-s` Random seed for reproducibility.
     - `-b` Column number (0-based) in BED files with methylation starts (e.g., 11 for m6A output from fibertools, 27 by default for full output).
     - `-o` Directory path for storing output files (models, list of reads used in training).
   - **Purpose:** Trains the model on a set of reads from specified datasets. Outputs the best model, a list of models in pickle format, and a TSV of reads used in training.
   - **Usage:** In general, run once per organism. Model parameters should remain consistent across similar datasets.

### 4. **Apply model**
   - **Command:** 
     - Single-core version: `apply_model.py -i path_to/dataset_1.bed -m path_to/best_model.pickle -t path_to/training-reads.tsv -g path_to/genome_name.h5 -p path_to/accessible_probs.tsv,path_to/inaccessible_probs.tsv -o path_to_output_directory`
     - Multiprocess version: `apply_model_multiprocess.py -i path_to/dataset.bed -m path_to/best_model.pickle -t path_to/training-reads.tsv -g path_to/genome_name.h5 -p path_to/accessible_probs.tsv,path_to/inaccessible_probs.tsv -o path_to_output_directory`
   - **Optional Parameters:**
     - `-l` Minimum footprints allowed per read (default: 0).
     - `-r` Enable circular mode (default: off).
     - `-s` Chunk size (default: 50000).
     - Multiprocess-specific:
       - `-c` Core count for parallel processing (default: all available CPU cores, typically I recommend 4-8 for stability).
       - `-x` Timeout in seconds before restarting the pool (default: core count * 100).
   - **Purpose:** Applies the trained model to remaining data. Outputs results in BED12 format, including footprint starts and lengths. Multiprocess version offers faster execution but may require tuning for stability.
   - **Note:** When using circular mode, reads are tiled 3x, affecting footprint count/length; further downstream processing is required.
