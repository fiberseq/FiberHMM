import pysam
import argparse
import random


"""
1) Correct aligned DddA BAM and replace likely deamination events with ambiguity codes (C|T: Y, G|A: R)
2) Create DA tag listing the molecular coordinates of deaminations
3) Add FD & LD tags for first and last deamination events in molecular coordinates

FIXED: Relaxed strand calling threshold for Nanopore data (was 0.90, now 0.51)
"""

# parse command line arguments
parser = argparse.ArgumentParser(description = "DddA BAM preprocessing",
    epilog = "")
parser.add_argument("-b", "--bam", required = True, metavar = '', help = "DddA aligned BAM to correct")
parser.add_argument("-c", "--cutoff", required = False, metavar = '', type=float, default=0.51,
                    help = "Strand mutation proportion cutoff (default: 0.51 for majority-wins)")
parser.add_argument("-r", "--random-strand", action="store_true",
                    help = "Randomly assign CT or GA strand when undetermined/none (off by default)")
args = parser.parse_args()

# identify fastq files in dir
bam_name = args.bam
cutoff = args.cutoff
random_strand = args.random_strand


def determine_da_strand_MD(read_obj, cutoff):
    """
    Based on the proportion of C->T & G->A determine the strand acted upon by DddA.
    Only counting single base substitutions.
    
    With cutoff=0.51 (majority-wins):
      - If more C->T than G->A: call CT strand
      - If more G->A than C->T: call GA strand
      - If tied or no mutations: undetermined/none
    """
    seq = read_obj.query_sequence
    pair = read_obj.get_aligned_pairs(matches_only=False, with_seq=True)
    c = 0
    g = 0
    total = 0
    for pos in pair:
        if pos[0] == None or pos[1] == None: # indel, ignore
            pass
        else:
            qi = pos[0]
            ref_pos = pos[2].upper()
            if seq[qi] != ref_pos:
                total += 1
                change = ref_pos + seq[qi]
                if change == "CT":
                    c += 1
                elif change == 'GA':
                    g += 1
    
    if c + g == 0:
        return 'none'
    
    ct_ratio = c / (c + g)
    ga_ratio = g / (c + g)
    
    # Majority wins (with cutoff threshold)
    if ct_ratio >= cutoff:
        return 'CT'
    elif ga_ratio >= cutoff:
        return 'GA'
    else:
        # If cutoff > 0.5 and neither passes, it's truly mixed
        # If cutoff <= 0.5, this shouldn't happen unless exactly tied
        return 'undetermined'


def check_num_assigned(sam_obj, cutoff):
    none = 0
    und = 0
    ct = 0
    ga = 0
    for read in sam_obj.fetch():
        if read.is_secondary == False and read.is_supplementary == False:
            change = determine_da_strand_MD(read, cutoff)
            if change == 'none':
                none += 1
            elif change == 'undetermined':
                und += 1
            elif change == 'CT':
                ct += 1
            else:
                ga += 1
    return({'CT':ct, 'GA':ga, 'Undetermined':und, 'None':none})


def correct_read_MD(read_obj, strand):
    """ Identify single-base changes from the reference that are likely induced by DddA
     and correct the original sequence using ambiguity codes (C|T: Y, G|A: R) and output new DA-tag positions.
     Limit detection to the previously identified DddA strand info (either ct or ga).
     Output everything in FIBER coordinates, not reference!
    """
    seq = read_obj.query_sequence
    pair = read_obj.get_aligned_pairs(matches_only=False, with_seq=True)
    new_seq = ''
    amb_codes = {'CT':'Y', 'GA':'R'}
    deam_pos = [] # mol coordinates of likely base changes
    for pos in pair:
        if pos[0] == None: # deletion, ignore
            pass
        elif pos[1] == None: # insertion, use seq base
            qi = pos[0]
            new_seq += seq[qi]
        else:
            qi = pos[0]
            ref_pos = pos[2].upper()
            if seq[qi] != ref_pos:
                change = ref_pos + seq[qi]
                if change == strand:
                    new_seq += amb_codes[change] # Update seq with ambiguity codes
                    deam_pos.append(qi+1) # track DA positions in 1-indexed
                else:
                    new_seq += seq[qi]
            else:
                new_seq += seq[qi]
    return(new_seq, deam_pos)


# write corrected reads to new BAM
bam = pysam.AlignmentFile(bam_name, "rb")
new_bam = bam_name.replace('.bam','_corrected.bam')
corrected_bam = pysam.AlignmentFile(new_bam, "wb", template=bam)

# Track stats
stats = {'CT': 0, 'GA': 0, 'undetermined': 0, 'none': 0}
random_assigned = {'CT': 0, 'GA': 0}  # Track random assignments separately

for read in bam.fetch():
    if read.is_secondary == False and read.is_supplementary == False:
        strand = determine_da_strand_MD(read, cutoff)
        MD = read.get_tag('MD')
        stats[strand] = stats.get(strand, 0) + 1
        
        # If random assignment enabled and strand is unclear, pick randomly
        if random_strand and strand in ['undetermined', 'none']:
            strand = random.choice(['CT', 'GA'])
            random_assigned[strand] += 1
        
        if strand in ['CT','GA']:
            # WRITE NEW seq with added tags
            new_seq, deam_pos = correct_read_MD(read, strand)
            if len(deam_pos) > 0:
                quals = read.query_qualities
                read.query_sequence = new_seq
                read.seq = new_seq
                read.query_qualities = quals
                read.set_tags([('DA', deam_pos), ('FD', deam_pos[0], "i"), ('LD', deam_pos[-1], "i"), ('ST', strand), ('MD', MD)])
            else:
                # Strand assigned but no deaminations found (edge case)
                read.set_tags([('DA', [0]), ('FD', 0, "i"), ('LD', 0, "i"), ('ST', strand), ('MD', MD)])
        else:
            read.set_tags([('DA', [0]), ('FD', 0, "i"), ('LD', 0, "i"), ('ST', strand), ('MD', MD)])
        corrected_bam.write(read)

bam.close()
corrected_bam.close()

# Print summary
print(f"Strand assignment summary (cutoff={cutoff}):")
print(f"  CT strand: {stats.get('CT', 0)}")
print(f"  GA strand: {stats.get('GA', 0)}")
print(f"  Undetermined: {stats.get('undetermined', 0)}")
print(f"  No mutations: {stats.get('none', 0)}")
print(f"  Total: {sum(stats.values())}")

if random_strand:
    print(f"\nRandom strand assignment enabled:")
    print(f"  Randomly assigned to CT: {random_assigned['CT']}")
    print(f"  Randomly assigned to GA: {random_assigned['GA']}")
    print(f"  Total randomly assigned: {sum(random_assigned.values())}")