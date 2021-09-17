import os
import argparse
import numpy as np
import pysam
import pysamstats
import h5py
from scipy.stats import mode


def process_dataseq(seq, transform):
    """
    Optionally transform data sequence, ensure both ends are 0, and return it.
    """
    if transform == 'ln1p':
        seq = np.log1p(seq)
    seq[0] = 0
    seq[-1] = 0
    return seq


def generate_hdf(bam_file, sizes_file, hdf_file, test_chr, test_str, transform):
    """
    Calculate transformed coverage values from BAM file and dump in HDF file for SeqerHMM to read.
    """
    # prepare all files
    bam = pysam.AlignmentFile(bam_file, 'rb', threads=os.cpu_count())
    with open(sizes_file, 'rt') as f:
        chr_lens = dict([[line.split('\t')[0], int(line.split('\t')[1].rstrip('\n'))] for line in f])
    hf = h5py.File(hdf_file, 'w')
    if test_chr == None:
        test_chr = list(bam.references)
    elif isinstance(test_chr, str) and test_chr.endswith('.txt'):
        with open(test_chr, 'rt') as f:
            test_chr = [line.rstrip('\n') for line in f]
    test_chr.sort()

    # read BAM, compute coverage, and write to HDF
    for chrom in test_chr:
        print('Processing chromosome {} of length {}'.format(chrom, chr_lens[chrom]))
        hf.create_group(chrom)
        pileup_it = pysamstats.stat_coverage_strand(bam, chrom=chrom, pad=True, max_depth=5e7, no_dup=True)
        if test_str == '.':
            pileup = np.array([rec['reads_all'] for rec in pileup_it])
            hf[chrom].create_dataset('{}.'.format(chrom), data=process_dataseq(pileup, transform))
        else:
            pileup = np.empty((2, chr_lens[chrom]))
            for i,rec in enumerate(pileup_it):
                if '+' in test_str:
                    pileup[0][i] = rec['reads_fwd']
                if '-' in test_str:
                    pileup[1][i] = rec['reads_rev']
            if '+' in test_str:
                hf[chrom].create_dataset('{}+'.format(chrom), data=process_dataseq(pileup[int(test_str == '-+')], transform))
            if '-' in test_str:
                hf[chrom].create_dataset('{}-'.format(chrom), data=process_dataseq(pileup[int(test_str != '-+')], transform))

    # write BAM information to text file with same location and header as the HDF file
    read_length = int(round(np.mean([read.query_length for read in bam.head(1000)])))
    n_reads = bam.count(read_callback='all')
    with open(hdf_file[:hdf_file.rfind('.')]+'_baminfo.txt', 'wt') as f:
        f.write('read_length\t{}\n'.format(read_length))
        f.write('n_reads\t{}\n'.format(n_reads))

    bam.close()
    hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HDF for SeqerHMM.')
    parser.add_argument('BAM_file', help='path to the .bam file')
    parser.add_argument('sizes_file', help='path to the .chrom.sizes file')
    parser.add_argument('HDF_file', help='path to the .hf, .hdf, .h5, etc. file to write coverage values to')
    chrom_subset = parser.add_mutually_exclusive_group()
    chrom_subset.add_argument('--chroms', nargs='+', help='subset of chromosomes to generate coverage for; e.g., '
                                                           '--chroms 1 2 6 VII X')
    chrom_subset.add_argument('--chroms_file', help='newline-delimited .txt file with one chromosome per line')
    parser.add_argument('--strands', default='.', choices=['.','+-','-+','+','-'], help='strandedness;\n'
                                                                                        '\'.\' for unstranded coverage;\n'
                                                                                        '\'+-\' for stranded analysis;\n'
                                                                                        '\'-+\' to flip strands (must be called as --strands=\'-+\');\n'
                                                                                        '\'+\' or \'-\' for single-strand analysis')
    parser.add_argument('--transform', default='ln1p', choices=['ln1p','none'], help='transform to apply to raw read counts; '
                                                                                       'must be either \'ln1p\' or \'none\'')
    args = parser.parse_args()

    test_chr = None
    if hasattr(args, 'chroms'):
        test_chr = args.chroms
    elif hasattr(args, 'chroms_file'):
        test_str = args.chroms_file

    generate_hdf(args.BAM_file, args.sizes_file, args.HDF_file, test_chr, args.strands, args.transform)
