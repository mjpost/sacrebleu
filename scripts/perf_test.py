#!/usr/bin/env python
import sys
import time
import statistics

sys.path.insert(0, '.')

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


N_REPEATS = 5


sys_files = [
    'data/wmt17-submitted-data/txt/system-outputs/newstest2017/cs-en/newstest2017.PJATK.4760.cs-en',
    'data/wmt17-submitted-data/txt/system-outputs/newstest2017/cs-en/newstest2017.uedin-nmt.4955.cs-en',
    'data/wmt17-submitted-data/txt/system-outputs/newstest2017/cs-en/newstest2017.online-A.0.cs-en',
    'data/wmt17-submitted-data/txt/system-outputs/newstest2017/cs-en/newstest2017.online-B.0.cs-en',
]

ref_files = ['data/wmt17-submitted-data/txt/references/newstest2017-csen-ref.en']

metrics = [
    # BLEU
    (BLEU, {}),
    (BLEU, {'tokenize': 'intl'}),
    (BLEU, {'tokenize': 'none', 'force': True}),
    # CHRF
    (CHRF, {}),
    (CHRF, {'whitespace': True}),
    # CHRF++
    (CHRF, {'word_order': 2}),
]


def create_metric(klass, kwargs, refs=None):
    if refs:
        # caching mode
        kwargs['references'] = refs
    return klass(**kwargs)


def read_files(*args):
    lines = []
    for fname in args:
        cur_lines = []
        with open(fname) as f:
            for line in f:
                cur_lines.append(line.strip())
        lines.append(cur_lines)
    return lines


def measure(metric_klass, metric_kwargs, systems, refs, cache=False):
    scores = []
    durations = []

    if cache:
        # caching mode
        metric_kwargs['references'] = refs
    st = time.time()
    metric = metric_klass(**metric_kwargs)

    for system in systems:
        sc = metric.corpus_score(system, None if cache else refs).score
        dur = time.time() - st
        print(f'{dur:.3f}', end=' ')
        durations.append(dur)
        scores.append(sc)
        st = time.time()

    durations = sorted(durations)
    median = durations[len(durations) // 2]
    std = statistics.pstdev(durations)
    mean = sum(durations) / len(durations)
    print(f' || mean: {mean:.3f} -- median: {median:.3f} -- stdev: {std:.3f}')


if __name__ == '__main__':

    systems = read_files(*sys_files)
    refs = read_files(*ref_files)

    msg = f'SacreBLEU {sacrebleu.__version__} performance tests'
    print('-' * len(msg) + '\n' + msg + '\n' + '-' * len(msg))

    for klass, kwargs in metrics:
        print(klass.__name__, kwargs)

        print(' > [no-cache] ', end='')
        measure(klass, kwargs, systems, refs, cache=False)

        print(' >   [cached] ', end='')
        measure(klass, kwargs, systems, refs, cache=True)
