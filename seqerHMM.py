import time
import argparse
import numpy as np
import pandas as pd
import pomegranate as pg
import h5py
import csv
from math import ceil
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize_scalar

pd.options.mode.chained_assignment = None


def coverage_gen(hf):
    """
    Return iterator over coverage values from HDF, chromosome by chromosome.
    """
    return (np.array([np.array(hf[chrom][strand]) for str_i,strand in enumerate(hf[chrom].keys())]) for chrom in hf.keys())


def obsv_gen(hf):
    """
    Return iterator over coverage values from HDF, chromosome by chromosome, and flipping reverse strands to standardize orientation.
    """
    return (np.array([np.array(hf[chrom][strand])[::-2*str_i+1] for str_i,strand in enumerate(hf[chrom].keys())]) for chrom in hf.keys())


def batch_obsv_gen(hf, batch_size):
    """
    Iterate through coverage values from HDF by batches.
    """
    tot_ct, curr_ct, batch = 0, 0, []
    for chrom in hf.keys():
        batch.extend([np.array(hf[chrom][strand])[::-2*str_i+1] for str_i,strand in enumerate(hf[chrom].keys())])
        curr_ct += 1
        tot_ct += 1
        if curr_ct == batch_size or tot_ct == len(hf.keys()):
            yield batch
            curr_ct, batch = 0, []


def kernel_density(x, kde):
    """
    Return scalar result of sklearn's KernelDensity estimator for a single input x.
    """
    return kde.score_samples(np.array([[x]]))[0]


def process_pred_seq(pred_seq, min_lens):
    """
    Fill in segments in predicted state sequence deemed too short, and return new sequence.
    """
    pred_seq[0] = 0
    pred_seq[-1] = 0
    ends = np.nonzero(np.ediff1d(pred_seq))[0]
    intervals = pd.DataFrame({'state':['I','E']*int((ends.size+1)/2)+['I'], 'seg_lens':np.ediff1d(np.concatenate([[-1],ends,[pred_seq.size-1]])), 'start':np.concatenate([[0],ends+1]), 'end':np.concatenate([ends,[pred_seq.size-1]])})
    intervals['rm'] = pd.Series([min_lens[intervals['state'].loc[idx]] for idx in intervals.index])
    while (intervals.loc[1:intervals.index.size-2,'seg_lens'] <= intervals.loc[1:intervals.index.size-2,'rm']).any():
        idx_min = (intervals.loc[1:intervals.index.size-2,'seg_lens']).subtract(intervals.loc[1:intervals.index.size-2,'rm']).idxmin()
        new_begin = intervals.loc[idx_min-1,'start']
        new_end = intervals.loc[idx_min+1,'end']
        intervals.loc[idx_min] = pd.Series({'state':intervals['state'].loc[idx_min-1], 'seg_lens':new_end-new_begin, 'start':new_begin, 'end':new_end, 'rm':min_lens[intervals['state'].loc[idx_min-1]]})
        intervals = intervals.drop(idx_min-1).drop(idx_min+1).reset_index(drop=True)
    new_pred_seq = np.empty(pred_seq.size)
    for interval in intervals.itertuples(index=True):
        new_pred_seq[interval.start:interval.end+1] = 0 if interval.state == 'I' else 1
    return new_pred_seq


def offset_5p(cov, offsets_5p):
    """
    Return appropriate offset for 5' transcript ends based on average transcript coverage.
    """
    return offsets_5p[0] * cov + offsets_5p[1]


def offset_3p(cov, offsets_3p):
    """
    Return appropriate offset for 3' transcript ends based on average transcript coverage.
    """
    return offsets_3p[0] * cov + offsets_3p[1]


def seqerHMM(hdf_file, sizes_file, gtf_file, test_chr, test_str, kde_samples, kde_bw, thresh_min, thresh_max, batch_size,
             threads, train_alg, decode_alg, offsets_5p, offsets_3p, min_lens, v):
    """
    Run SeqerHMM and write results to GTF file.
    """
    start_time = time.perf_counter()
    old_time = start_time

    if v: print('Loading Data... ', end='')
    with open(sizes_file, 'rt') as f:
        chr_lens = dict([[line.split('\t')[0], int(line.split('\t')[1].rstrip('\n'))] for line in f])
    hf = h5py.File(hdf_file, 'r')
    with open(hdf_file[:hdf_file.rfind('.')]+'_baminfo.txt', 'rt') as f:
        read_length = int(next(f).split('\t')[1].rstrip('\n'))
        n_reads = int(next(f).split('\t')[1].rstrip('\n'))
    if test_chr == None:
        test_chr = list(hf.keys())
    elif isinstance(test_chr, str) and test_chr.endswith('.txt'):
        with open(test_chr, 'rt') as f:
            test_chr = [line.rstrip('\n') for line in f]
    test_chr.sort()
    if batch_size == -1:
        batch_size = len(hf.keys())
    curr_time = time.perf_counter()
    if v: print('{:.2f}s'.format(curr_time - old_time))
    old_time = curr_time

    if v: print('Running Threshold Model... ', end='')
    exp_sample = [np.random.choice(chrom_data.flatten(), size=round(kde_samples*(chr_lens[chrom]/sum(chr_lens.values())))) for chrom, chrom_data in zip(test_chr, coverage_gen(hf))]
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bw, rtol=1e-5).fit(np.concatenate(exp_sample).reshape((-1,1)))
    threshold = minimize_scalar(kernel_density, bounds=(thresh_min, thresh_max), args=(kde,)).x
    pred_seqs_thresh = [(chrom > threshold).astype('int') for chrom in coverage_gen(hf)]
    pred_tss_thresh = [[np.where((pred_seqs_str[:-1] == (0 if strand=='+' else 1)) & (pred_seqs_str[1:] == (1 if strand=='+' else 0)))[0] + (2 if strand=='+' else 1) for pred_seqs_str,strand in zip(pred_seqs_chr,test_str)] for pred_seqs_chr in pred_seqs_thresh]
    pred_tes_thresh = [[np.where((pred_seqs_str[:-1] == (1 if strand=='+' else 0)) & (pred_seqs_str[1:] == (0 if strand=='+' else 1)))[0] + (1 if strand=='+' else 2) for pred_seqs_str,strand in zip(pred_seqs_chr,test_str)] for pred_seqs_chr in pred_seqs_thresh]
    curr_time = time.perf_counter()
    if v:
        print('{:.2f}s'.format(curr_time-old_time))
        print('Threshold found at coverage value: {:.2f}'.format(threshold))
    old_time = curr_time

    if v: print('Building and Initializing HMM... ', end='')
    intergenic_param = np.average([pg.PoissonDistribution.from_samples(rchr[pstchr == 0]).parameters[0] for pstchr, rchr in zip(pred_seqs_thresh, coverage_gen(hf))], weights=[chrom[1] for chrom in sorted(chr_lens.items())])
    exon_params = np.average([pg.NormalDistribution.from_samples(rchr[pstchr == 1]).parameters for pstchr, rchr in zip(pred_seqs_thresh, coverage_gen(hf))], axis=0, weights=[[chrom[1],chrom[1]] for chrom in sorted(chr_lens.items())])
    intergenic = pg.State(pg.PoissonDistribution(intergenic_param) if intergenic_param > 0 else pg.PoissonDistribution(1e-9), name="s1")
    exon = pg.State(pg.NormalDistribution(exon_params[0], exon_params[1]), name="s2")

    up = np.mean([ptssstr.size/np.count_nonzero(pststr==0) for pstchr,ptsschr in zip(pred_seqs_thresh,pred_tss_thresh) for pststr,ptssstr in zip(pstchr,ptsschr)])
    down = np.mean([ptesstr.size/np.count_nonzero(pststr==1) for pstchr,pteschr in zip(pred_seqs_thresh,pred_tes_thresh) for pststr,ptesstr in zip(pstchr,pteschr)])

    exp_model = pg.HiddenMarkovModel()
    exp_model.add_states(intergenic, exon)
    exp_model.add_transition(exp_model.start, intergenic, 1.0)
    exp_model.add_transition(intergenic, intergenic, 1-up)
    exp_model.add_transition(intergenic, exon, up)
    exp_model.add_transition(exon, exon, 1-down)
    exp_model.add_transition(exon, intergenic, down)
    exp_model.add_transition(intergenic, exp_model.end, 1.0343e-6)
    exp_model.bake()
    if v: print('{:.2f}s'.format(time.perf_counter()-old_time))

    if v: print('Training HMM... \n', end='')
    obsv_seqs_exp_train = batch_obsv_gen(hf, batch_size)
    for batch_i in range(ceil(len(test_chr)/batch_size)):
        obsv_seqs_exp_batch = next(obsv_seqs_exp_train)
        if v: print('Training on batch {} of {} chromosomes'.format(batch_i+1, len(obsv_seqs_exp_batch)//2))
        exp_model.fit(obsv_seqs_exp_batch, algorithm=train_alg, verbose=v, n_jobs=threads)
    old_time = time.perf_counter()

    if v: print('Predicting and Processing Transcript Boundaries... ', end='')
    pred_seqs_exp = (np.array([exp_model.predict(obsv_seq_str, algorithm=decode_alg)[::-2*str_i+1] if decode_alg=='map' else exp_model.predict(obsv_seq_str, algorithm=decode_alg)[1-3*str_i+1:str_i-1:-2*str_i+1] for str_i,obsv_seq_str in enumerate(obsv_seq_chr)]) for obsv_seq_chr in obsv_gen(hf))
    pred_seqs_exp_p = ((process_pred_seq(pred_seq_str, min_lens) for pred_seq_str in pred_seq_chr) for pred_seq_chr in pred_seqs_exp)
    breakpoints = [[np.where(pred_seqs_str[:-1] != pred_seqs_str[1:])[0] + 1 for pred_seqs_str in pred_seqs_chr] for pred_seqs_chr in pred_seqs_exp_p]
    for bp_chr in breakpoints:
        for bp_str in bp_chr:
            bp_str[...][::2] += 1
    pred_exons = [[bp_str.reshape(-1,2) for str_i,bp_str in enumerate(bp_chr)] for bp_chr in breakpoints]
    pred_exon_covs = [[np.array([np.mean(cov_str[pe[0]-1:pe[1]]) for pe in pe_str]) for pe_str,cov_str in zip(pe_chr,cov_chr)] for pe_chr,cov_chr in zip(pred_exons,coverage_gen(hf))]
    processed_exons = [[np.stack((np.clip(pe_str[:,str_i]+(-2*str_i+1)*offset_5p(pec_str, offsets_5p), 1, chr_lens[chrom]), np.clip(pe_str[:,-1*str_i+1]+(-2*str_i+1)*offset_3p(pec_str, offsets_3p), 1, chr_lens[chrom]))[::-2*str_i+1], axis=1).astype('int') for str_i,(pe_str,pec_str) in enumerate(zip(pe_chr,pec_chr))] for pe_chr,pec_chr,chrom in zip(pred_exons, pred_exon_covs, test_chr)]
    curr_time = time.perf_counter()
    if v: print('{:.2f}s'.format(curr_time-old_time))
    old_time = curr_time

    if v: print('Estimating Expression... ', end='')

    fpkms = [[[np.sum(np.exp(cov_str[xon[0]-1:xon[1]])-1)*1e9/(read_length*(xon[1]-xon[0]+2)*n_reads) for xon in pe_str] for pe_str,cov_str in zip(pe_chr,cov_chr)] for pe_chr,cov_chr in zip(processed_exons,coverage_gen(hf))]
    total_fpkm = np.sum([fpkm for chrom in fpkms for strand in chrom for fpkm in strand])
    tpms = [[[fpkm*1e6/total_fpkm for fpkm in strand] for strand in chrom] for chrom in fpkms]
    curr_time = time.perf_counter()
    if v: print('{:.2f}s'.format(curr_time-old_time))
    old_time = curr_time

    hf.close()

    if v: print('Writing Results... ', end='')
    with open(gtf_file, 'wt', newline='') as f:
        f.write('# SeqerHMM GTF output\n')
        f.write('# Model parameters:\n')
        f.write('# HDF_file {} sizes_file {} GTF_file {} test_chr {} test_str {} KDE_samples {} KDE_bw {} thresh_min {}'
                'thresh_max {} batch_size {} train_alg {} decode_alg {} min_lens {} offsets_5p {} offsets_3p {}\n'
                .format(hdf_file, sizes_file, gtf_file, test_chr, test_str, kde_samples, kde_bw, thresh_min, thresh_max,
                        batch_size, train_alg, decode_alg, min_lens, offsets_5p, offsets_3p))
        gtf_writer = csv.writer(f, delimiter='\t')
        for chr_i, chrom in enumerate(test_chr):
            for str_i, strand in enumerate(test_str):
                for xon_i, (xon, fpkm, tpm) in enumerate(zip(processed_exons[chr_i][str_i], fpkms[chr_i][str_i], tpms[chr_i][str_i])):
                    gtf_writer.writerow([chrom, 'SeqerHMM', 'exon', xon[0], xon[1], '1000', strand, '.',
                                         '{} {} {} {} {}'.format('gene_id "SQRHMM.{}";'.format(xon_i+1), 'transcript_id "SQRHMM.{}.1";'.format(xon_i+1), 'exon_number "1";', 'FPKM "{}";'.format(fpkm), 'TPM "{}";'.format(tpm))])
    curr_time = time.perf_counter()
    if v:
        print('{:.2f}s'.format(curr_time-old_time))
        print('Done. Total time: {:.2f}s'.format(curr_time-start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeqerHMM.')
    parser.add_argument('HDF_file', help='path to the .hf, .hdf, .h5, etc. file to read coverage values from')
    parser.add_argument('sizes_file', help='path to the .chrom.sizes file')
    parser.add_argument('GTF_file', help='path to the .gtf file to write predicted transcripts to')
    chrom_subset = parser.add_mutually_exclusive_group()
    chrom_subset.add_argument('--chroms', nargs='+', help='subset of chromosomes to generate coverage for; e.g., '
                                                           '--chroms 1 2 6 VII X')
    chrom_subset.add_argument('--chroms_file', help='newline-delimited .txt file with one chromosome per line')
    parser.add_argument('--strands', default='.', choices=['.','+-','+','-'], help='strandedness;\n'
                                                                                   '\'.\' for unstranded coverage;\n'
                                                                                   '\'+-\' for stranded analysis;\n'
                                                                                   '\'+\' or \'-\' for single-strand analysis')
    parser.add_argument('--KDE_samples', type=int, default=100000, help='number of coverage values to sample for running kernel density estimation')
    parser.add_argument('--KDE_bw', type=float, default=0.25, help='bandwidth to use for kernel density estimation')
    parser.add_argument('--thresh_min', type=float, default=0, help='minimum bound of estimated threshold')
    parser.add_argument('--thresh_max', type=float, default=5, help='maximum bound of estimated threshold')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size (in number of chromosomes) for HMM training')
    parser.add_argument('--threads', type=int, default=-1, help='number of parallel threads to use for HMM training')
    parser.add_argument('--train_alg', default='baum-welch', choices=['baum-welch','viterbi'], help='algorithm to use for HMM training')
    parser.add_argument('--decode_alg', default='map', choices=['map','viterbi'], help='algorithm to use for HMM decode_alg')
    parser.add_argument('--offsets_5p', type=float, nargs=2, default=[0,0], help='coefficient m and intercept b of the linear offset function'
                                                                          'bp = m * coverage + b for 5\' transcript ends')
    parser.add_argument('--offsets_3p', type=float, nargs=2, default=[0,50], help='coefficient m and intercept b of the linear offset function'
                                                                          'bp = m * coverage + b for 3\' transcript ends')
    parser.add_argument('--min_lens', type=int, nargs=2, default=[100, 200], help='the minimum intertranscript and transcript lengths, respectively')
    parser.add_argument('--v', action='store_true', help='verbose output')
    args = parser.parse_args()
    min_lens = {'I':args.min_lens[0], 'E':args.min_lens[1]}

    test_chr = None
    if hasattr(args, 'chroms'):
        test_chr = args.chroms
    elif hasattr(args, 'chroms_file'):
        test_str = args.chroms_file
    test_str = [c for c in args.strands]

    seqerHMM(args.HDF_file, args.sizes_file, args.GTF_file, test_chr, test_str, args.KDE_samples, args.KDE_bw,
             args.thresh_min, args.thresh_max, args.batch_size, args.threads, args.train_alg, args.decode_alg,
             args.offsets_5p, args.offsets_3p, min_lens, args.v)
