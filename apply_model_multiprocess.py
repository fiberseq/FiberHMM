import pandas as pd
import numpy as np
import getopt
import sys
from hmmlearn import hmm
import os
import pickle
import tempfile
import glob
import h5py
import time
import logging
from multiprocessing import Pool, cpu_count, Manager
from threading import Thread
import queue

pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def options():
    # default parameters
    me_col = 27
    chunk_size = 10000
    train_rids = []
    min_len = 0
    core_count = cpu_count()
    edge_trim = 10
    #typically on my test machine this runs at 40-60 iterations/second, so this provides a very large buffer
    timeout = core_count*100
    circle=False

    optlist, args = getopt.getopt(sys.argv[1:], 'i:g:m:t:o:b:s:l:c:x:re:',
                                  ['infile=', 'context=', 'model=', 'train_reads=', 'outdir=', 'me_col=', 'min_len=', 'core_count=', 'timeout=', 'circular', 'edge_trim='])

    for o, a in optlist:
        if o == '-i' or o == '--infile':
            infile = a
        elif o == '-g' or o == '--context':
            context = a
        elif o == '-m' or o == '--model':
            model = a
        elif o == '-t' or o == '--train_reads':
            train_rids = pd.read_csv(a)
            train_rids = train_rids['rid'].tolist()
        elif o == '-o' or o == '--outdir':
            outdir = a
        elif o == '-b' or o == '--me_col':
            me_col = int(a)
        elif o == '-s' or o == '--chunk_size':
            chunk_size = int(a)
        elif o == '-l' or o == '--min_len':
            min_len = int(a)
        elif o == '-c' or o == '--core_count':
            core_count = int(a)
        elif o == '-x' or o == '--timeout':
            timeout = int(a)
        elif o == '-r' or o == '--circular':
            circle = True
        elif o == '-e' or o == '--edge_trim':
            edge_trim = int(a)

    logging.info("Options parsed successfully.")
    return infile, context, model, train_rids, outdir, me_col, chunk_size, min_len, core_count, timeout, circle, edge_trim


def encode_me(rid, read, read_info, context, circle, edge_trim):
    #grab info, read
    chrom = read_info.loc[rid]['chrom']
    start = read_info.loc[rid, 'start']
    end = read_info.loc[rid, 'end']

    me = np.array(read.dropna()).astype(int)
    #remove any methylations in the trim region
    me = me[np.where(
        (me>edge_trim)&(me<((end-start)-edge_trim))
        )] 

    #make sure within range, find positions with no methylation
    me = me[np.where(me < (end - start))[0]]
    no_me = np.arange(end - start)
    no_me = np.delete(no_me, me)
   
    #grab sequence context info from context file 
    with h5py.File(context, 'r', swmr=True) as f:
        hexamers = f[chrom]['table'][(start+edge_trim):(end-edge_trim)]

    #encode methylations and no methylations
    me_encode = np.array([item[1] for item in hexamers])
    #add non-A (so, 0% probability of methylation) to edge bases
    me_encode = np.pad(me_encode, pad_width=((edge_trim, edge_trim), (0, 0)), mode='constant', constant_values=4096)
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


def process_chunk(chunk, model, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, dataset, i, circle, edge_trim):
    log_message = f"Processing chunk {i}..."
    sys.stdout.write(f'\r{log_message}')
    sys.stdout.flush()

    try:
        # Make sure chromosomes are in context file
        chunk = chunk.loc[chunk['chrom'].isin(chromlist)]

        # Remove reads used in training if provided
        chunk = chunk.loc[~chunk['rid'].isin(train_rids)]

        # Generate bed12 for reads with no methylation
        no_me_b12 = chunk.loc[chunk['me'] == '.'].drop('me', axis=1)
        no_me_b12['thickStart'] = no_me_b12['start']
        no_me_b12['thickEnd'] = no_me_b12['end']
        no_me_b12['itemRgb'] = '255,0,0'
        no_me_b12['blockCount'] = 1
        no_me_b12['blockStarts'] = 1
        if not circle:
            no_me_b12['blockSizes'] = no_me_b12['end'] - no_me_b12['start']
        else:
            no_me_b12['blockSizes'] = no_me_b12['end']*3 - no_me_b12['start']

        no_me_b12.columns = ['chrom', 'start', 'end', 'name', 'thickStart', 'thickEnd', 'blockCount', 'itemRgb', 'blockSizes', 'blockStarts']

        # Generate bed12 for reads with methylation
        chunk = chunk.loc[chunk['me'] != '.']
        b12 = chunk.drop(columns=['me'])
       # b12['start'] += edge_trim
       # b12['end'] -= edge_trim

        b12['thickStart'] = b12['start']
        b12['thickEnd'] = b12['end']
        b12['itemRgb'] = '255,0,0'

        # Grab methylations
        chunk = chunk['me'].str.split(pat=',', expand=True)

        all_starts = []
        all_lengths = []
        all_counts = []

        # Encode methylations, predict footprints
        for index, read in chunk.iterrows():
            read_encode = encode_me(index, read, b12, context, circle, edge_trim)
            read_encode = read_encode.astype(int).reshape(-1, 1)
            pos = model.predict(read_encode)

            #if no footprints, add '.'
            if sum(pos) == 0:
                all_starts.append('.')
                all_lengths.append('.')
                all_counts.append(0)

            #else, find changepoints and lengths
            else:
                pos_offset = np.append([0], pos)
                pos = np.append(pos, [0])
                pos_diff = pos_offset - pos
                starts = np.where(pos_diff == -1)[0]
                ends = np.where(pos_diff == 1)[0]
                ends = np.append([0], ends)
                lengths = np.sum((starts * -1, ends[1:]), axis=0).astype('int')

                #add to list for final dataframe
                all_starts.append(','.join(starts.astype(str)))
                all_lengths.append(','.join(lengths.astype(str)))
                all_counts.append(len(starts))

        # Add to bed12
        b12['blockCount'] = all_counts
        b12['blockStarts'] = all_starts
        b12['blockSizes'] = all_lengths

        # Combine, sort bed12s
        b12 = b12.rename(columns={'rid': 'name'})
        b12.columns = ['chrom', 'start', 'end', 'name', 'thickStart', 'thickEnd', 'blockCount', 'itemRgb', 'blockStarts', 'blockSizes']
        b12 = pd.concat([b12, no_me_b12])
        b12 = b12.sort_values(by=['chrom', 'start'])

        # Write to a temporary file 
        tmp_file = os.path.join(tmp_dir, f"{dataset}_{i}.bed")
        b12.to_csv(tmp_file, sep='\t', index=False, header=False)

        log_message = f"Chunk {i} processed successfully."
        sys.stdout.write(f'\r{log_message}\n')
        sys.stdout.flush()
    except Exception as e:
        log_message = f"Error processing chunk {i}: {e}"
        sys.stdout.write(f'\r{log_message}\n')
        sys.stdout.flush()
        return False  # Signal failure
    return True  # Signal success


def monitor_tasks(tasks, results_queue, timeout):
    #monitor tasks and restart any that exceed the timeout. Might help if script hangs, although I'm not sure it does
    while tasks:
        task_id, task = tasks.pop(0)
        try:
            result = task.get(timeout=timeout)
            results_queue.put((task_id, result))
        except queue.Empty:
            # Timeout exceeded, restarting the task
            results_queue.put((task_id, False))  # Mark this task as failed
            logging.warning(f"Task {task_id} timed out and will be restarted.")
            tasks.append((task_id, task))  # Requeue the task to try again


def apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, core_count, timeout, circle, edge_trim):
    dataset = f.split('/')[-1].split('.')[0]
    logging.info(f"Starting model application for dataset {dataset}.")

    manager = Manager()
    results_queue = manager.Queue()
    tasks = []
    
    try:
        # read in fibertools output bedfile in chunks
        reader = pd.read_csv(f, usecols=[0, 1, 2, 3, me_col], names=['chrom', 'start', 'end', 'rid', 'me'], sep='\t', comment='#', chunksize=chunk_size)

        #assign each chunk to a pool
        with Pool(core_count) as pool:
            for i, chunk in enumerate(reader):
                task = pool.apply_async(process_chunk, args=(chunk, model, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, dataset, i, circle, edge_trim))
                tasks.append((i, task))

            monitor = Thread(target=monitor_tasks, args=(tasks, results_queue, timeout))
            monitor.start()
            monitor.join()  # Wait for all tasks to be monitored

        logging.info(f"Model application for dataset {dataset} completed successfully.")
    except Exception as e:
        logging.error(f"Error applying model to dataset {dataset}: {e}", exc_info=True)


def combine_temp_files(chromlist, tmp_dir, outdir, dataset):
    #combine the tempfiles into a single large bedfile
    #this consistently fails because of permissions, but it's not a huge issue
    logging.info(f"Combining temporary files for dataset {dataset}.")

    try:
        tmp_files = glob.glob(os.path.join(tmp_dir, f"{dataset}_*.bed"))
        final_file = os.path.join(outdir, f"{dataset}_fp.bed")
        with open(final_file, 'w') as fout:
            for tmp_file in tmp_files:
                with open(tmp_file, 'r') as fin:
                    fout.write(fin.read())
                os.remove(tmp_file)
        
        logging.info(f"Temporary files for dataset {dataset} combined successfully.")
    except Exception as e:
        logging.error(f"Error combining temporary files for dataset {dataset}: {e}", exc_info=True)


f, context, model, train_rids, outdir, me_col, chunk_size, min_len, core_count, timeout, circle, edge_trim = options()

dataset = f.split('/')[-1].split('.')[0]

# Grab list of chromosomes from context file
with pd.HDFStore(context, 'r') as tmp:
    chromlist = np.array(tmp.keys())
chromlist = np.array([s[1:] for s in chromlist])
chromlist = chromlist[~np.char.find(chromlist, '_') >= 0]

# Make output directory and temporary directory
if not os.path.exists(outdir):
    os.mkdir(outdir)
tmp_dir = tempfile.mkdtemp(dir=outdir)

# Load model
with open(model, 'rb') as handle:
    model = pickle.load(handle)
logging.info(f"Model loaded successfully.")

# Run model
apply_model(model, f, outdir, context, chromlist, train_rids, me_col, chunk_size, min_len, tmp_dir, core_count, timeout, circle, edge_trim)

# Combine temp files, remove
combine_temp_files(chromlist, tmp_dir, outdir, dataset)

#this consistently fails on my tests because of permissions, but it's not a huge issue
os.rmdir(tmp_dir)
logging.info("Temporary directory removed and script completed.")
