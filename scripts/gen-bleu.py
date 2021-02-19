#!/usr/bin/env python

import subprocess

SYSTEMS = [
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-cs/newstest2017.online-A.0.en-cs.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-cs/newstest2017.PJATK.4761.en-cs.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/cs-en/newstest2017.PJATK.4760.cs-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/cs-en/newstest2017.online-A.0.cs-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-ru/newstest2017.online-A.0.en-ru.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/ru-en/newstest2017.uedin-nmt.4890.ru-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/zh-en/newstest2017.xmunmt.5160.zh-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-de/newstest2017.LIUM-NMT.4900.en-de.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-lv/newstest2017.tilde-nc-nmt-smt-hybrid.5047.en-lv.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/fi-en/newstest2017.TALP-UPC.4937.fi-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/tr-en/newstest2017.LIUM-NMT.4888.tr-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-zh/newstest2017.SogouKnowing-nmt.5131.en-zh.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-tr/newstest2017.online-B.0.en-tr.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/en-fi/newstest2017.apertium-unconstrained.4769.en-fi.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/lv-en/newstest2017.tilde-c-nmt-smt-hybrid.5051.lv-en.sgm',
    'wmt17-submitted-data/sgm/system-outputs/newstest2017/de-en/newstest2017.RWTH-nmt-ensemble.4920.de-en.sgm',
]

mteval_14, mteval_14_intl, sacrebleu_13a, sacrebleu_intl = {}, {}, {}, {}



if __name__ == '__main__':
    print('declare -A MTEVAL14=( ', end='')
    for system in SYSTEMS:
        system = 'data/' + system
        parts = system.split('/')
        sys_file = parts[-1]
        test_set = sys_file.split('.')[0]
        lpair = parts[-2]
        sl, tl = lpair.split('-')
        ref_path = '/'.join(parts[:-3]).replace('system-outputs', 'references')
        ref_path = f'{ref_path}/{test_set}-{sl}{tl}-ref.{tl}.sgm'

        src_path = ref_path.replace('references', 'sources')
        src_path = src_path.replace(f'ref.{tl}', f'src.{sl}')

#         txt_system = system.replace('.sgm', '').replace('sgm', 'txt')
        # print(f'{sys_file}')

        # foo = subprocess.check_output([
            # 'scripts/mteval-v14.pl', '-c', '-b',
            # '-s', src_path, '-r', ref_path, '-t', system],
            # universal_newlines=True)

        # bleu = float(foo.strip())
        # mteval_14[sys_file] = bleu
        # print(f' mteval-v14a.pl -c        => {bleu:.4f}')

        # foo = subprocess.check_output([
            # 'sacrebleu', '-w', '4', '-t', 'wmt17', '-l', f'{sl}-{tl}',
            # '--tokenize', '13a', '-b', '-i', txt_system],
            # universal_newlines=True)
        # bleu = float(foo.strip())
        # print(f' sacrebleu --tok 13a      => {bleu:.4f}')
        # sacrebleu_13a[sys_file] = bleu


        foo = subprocess.check_output([
            'scripts/mteval-v14.pl', '-c', '-b', '--international-tokenization',
            '-s', src_path, '-r', ref_path, '-t', system],
            universal_newlines=True)
        bleu = float(foo.strip())
        print(f'["{sys_file}"]={bleu:.2f}')

        # mteval_14_intl[sys_file] = bleu
        # #print(f' mteval-v14a.pl -c [intl] => {bleu:.4f}')


        # foo = subprocess.check_output([
            # 'sacrebleu', '-w', '4', '-t', 'wmt17', '-l', f'{sl}-{tl}',
            # '--tokenize', 'intl', '-b', '-i', txt_system],
            # universal_newlines=True)
        # bleu = float(foo.strip())
        # #print(f' sacrebleu --tok intl     => {bleu:.4f}')
        # sacrebleu_intl[sys_file] = bleu

    print(')')
