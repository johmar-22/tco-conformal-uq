# -*- coding: utf-8 -*-
"""
decode_soap_index.py -- mechanical decoding of SOAP power-spectrum feature
indices for the NOMAD 2018 TCO study.

Maps a flat feature name (e.g. 'Max_0039', 'Std_3877') or a bare integer index
(0..5993) to its (species pair, n, n', l) assignment, using DScribe's own
soap.get_location() so the decoding is mechanically tied to the descriptor
implementation rather than derived by hand.

Configuration matches the study pipeline exactly:
    species = sorted({'Al', 'Ga', 'In', 'O'}), periodic=True,
    r_cut = 6.0, n_max = 9, l_max = 8

Key facts (verified against DScribe 2.1.2):
  * DScribe sorts species internally by atomic number (O=8, Al=13, Ga=31,
    In=49), regardless of the input order. Block layout:
      O-O   [   0,  405)    Al-O  [ 405, 1134)    Ga-O  [1134, 1863)
      In-O  [1863, 2592)    Al-Al [2592, 2997)    Al-Ga [2997, 3726)
      Al-In [3726, 4455)    Ga-Ga [4455, 4860)    Ga-In [4860, 5589)
      In-In [5589, 5994)
  * Within each block, l is the OUTERMOST index (contiguous sub-blocks
    l = 0..l_max); within each l sub-block, (n, n') is enumerated row-major,
    with n' >= n for same-species pairs (upper triangular).
  * Same-species blocks: n_max(n_max+1)/2 x (l_max+1) = 45 x 9 = 405 features.
    Cross-species blocks: n_max^2 x (l_max+1) = 81 x 9 = 729 features.
    Total: 4 x 405 + 6 x 729 = 5,994 per aggregation statistic.

Usage:
    python decode_soap_index.py Max_0039 Std_3877
    python decode_soap_index.py 39 3877 --blocks
"""
import argparse
import sys

N_MAX, L_MAX = 9, 8
SPECIES = sorted({'Al', 'Ga', 'In', 'O'})

def get_soap():
    from dscribe.descriptors import SOAP
    return SOAP(species=SPECIES, periodic=True, r_cut=6.0,
                n_max=N_MAX, l_max=L_MAX, sparse=False)

def pair_list(soap):
    # atomic-number order, as used internally by DScribe
    z = {'O': 8, 'Al': 13, 'Ga': 31, 'In': 49}
    sp = sorted(SPECIES, key=lambda s: z[s])
    return [(sp[i], sp[j]) for i in range(len(sp)) for j in range(i, len(sp))]

def decode(gidx, soap, pairs):
    for p in pairs:
        loc = soap.get_location(p)
        if loc.start <= gidx < loc.stop:
            off  = gidx - loc.start
            same = p[0] == p[1]
            per_l = N_MAX * (N_MAX + 1) // 2 if same else N_MAX * N_MAX
            l, r = divmod(off, per_l)
            if same:
                n = 0
                while r >= N_MAX - n:
                    r -= N_MAX - n
                    n += 1
                npr = n + r
            else:
                n, npr = divmod(r, N_MAX)
            return dict(index=gidx, pair=f'{p[0]}-{p[1]}',
                        block=f'[{loc.start}, {loc.stop})',
                        n=n + 1, n_prime=npr + 1, l=l)   # 1-based n, n'
    raise ValueError(f'index {gidx} outside [0, {soap.get_number_of_features()})')

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('features', nargs='+',
                    help="feature names (Max_0039) or bare indices (39)")
    ap.add_argument('--blocks', action='store_true',
                    help='also print the full block layout')
    args = ap.parse_args()

    import importlib.metadata
    print(f"DScribe version: {importlib.metadata.version('dscribe')}")
    soap  = get_soap()
    pairs = pair_list(soap)
    assert soap.get_number_of_features() == 5994

    if args.blocks:
        print('\nBlock layout (atomic-number-sorted pairs):')
        for p in pairs:
            loc = soap.get_location(p)
            print(f'  {p[0]:>2}-{p[1]:<2}  [{loc.start:>4}, {loc.stop:>4})')

    print()
    for f in args.features:
        if '_' in f:
            agg, idx = f.split('_')
            d = decode(int(idx), soap, pairs)
            print(f"{f}: aggregation={agg}, pair={d['pair']}, "
                  f"(n, n') = ({d['n']}, {d['n_prime']}), l = {d['l']}   "
                  f"[block {d['block']}]")
        else:
            d = decode(int(f), soap, pairs)
            print(f"index {f}: pair={d['pair']}, (n, n') = ({d['n']}, "
                  f"{d['n_prime']}), l = {d['l']}   [block {d['block']}]")

if __name__ == '__main__':
    main()
