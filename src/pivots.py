'''
Select a set of pivot points from a source embedding
set and a target vocabulary, using one of two schemes:

- Frequent : take the top N shared words by frequency in
             target corpus
- Random : take N random shared words

Pivot items are written, one per line, to a file.
The output file also included commented (#) headers
logging the configuration that generated it.
'''

import codecs
import random
from datetime import datetime
import pyemblib

def readStopwords(f, tolower=False):
    stops = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            word = line.strip()
            if tolower: word = word.lower()
            stops.add(word)
    return stops

def readVocab(f, tolower=False, filter_to=None):
    
    if filter_to:
        filter_set = set(filter_to)
        key_filter = lambda k: k in filter_set
    else:
        key_filter = lambda k: True

    vocab = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (key, freq) = [s.strip() for s in line.split()]
            if key_filter(key):
                if tolower: key = key.lower()
                vocab[key] = int(freq)
    return vocab

def frequentPivotTerms(src_vocab, trg_vocab, num_terms=-1, stopwords=None):
    
    # sort target vocab by frequency
    trg_pairs = [(key, value) for (key, value) in trg_vocab.items()]
    trg_pairs.sort(key=lambda k:k[1], reverse=True)

    i, pivots = 0, []
    while i < len(trg_pairs) and (num_terms < 0 or len(pivots) < num_terms):
        (word, freq) = trg_pairs[i]
        if (stopwords is None) or (not word in stopwords):
            if word in src_vocab:
                pivots.append(word)
        i += 1

    return pivots

def randomPivotTerms(src_vocab, trg_vocab, num_terms=-1, stopwords=None):
    shared = list(src_vocab.intersection(trg_vocab))
    random.shuffle(shared)

    pivots, i = [], 0
    while i < len(shared) and (num_terms < 0 or len(pivots) < num_terms):
        if (stopwords is None) or (not shared[i] in stopwords):
            pivots.append(shared[i])
        i += 1

    return pivots

def writePivotsFile(pivots, options):
    with codecs.open(options.outf, 'w', 'utf-8') as stream:
        # write headers
        for header in [
                'Generated: %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Mode: %s' % ('frequent' if options.frequent else 'random'),
                'Number of pivots: %s' % (str(options.num_pivots) if options.num_pivots > 0 else 'all shared words'),
                'Source embeddings: %s' % options.src_embf,
                'Target embeddings: %s' % (options.trg_embf if options.trg_embf else '-- none supplied --'),
                'Target vocabulary: %s' % (options.trg_vocabf if options.trg_vocabf else '-- none supplied --'),
                ''
            ]:
            stream.write('# %s\n' % header)

        # write pivot terms
        for pivot in pivots:
            stream.write('%s\n' % pivot)

def readPivotsFile(f, tolower=False):
    pivots = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        in_headers = True
        for line in stream:
            # skip the headers
            if in_headers and line[0] != '#':
                in_headers = False
            if not in_headers:
                term = line.strip()
                if tolower: term = term.lower()
                pivots.add(term)
    return pivots

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('--src-embeddings', dest='src_embf',
                help='source embeddings file (REQUIRED)')
        parser.add_option('--src-embeddings-mode', dest='src_embf_mode',
                help='source embeddings file mode (default: %default)',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text)
        parser.add_option('--trg-embeddings', dest='trg_embf',
                help='target embeddings file (must supply --trg-embeddings or --trg-vocab)')
        parser.add_option('--trg-embeddings-mode', dest='trg_embf_mode',
                help='target embeddings file mode (default: %default)',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text)
        parser.add_option('--trg-vocab', dest='trg_vocabf',
                help='target vocabulary frequencies file (must supply --trg-embeddings or --trg-vocab)')
        parser.add_option('--output', dest='outf',
                help='file to write pivot terms to (REQUIRED)')
        parser.add_option('--stopwords', dest='stopwordf',
                help='file with stopwords to ignore as potential pivots')
        parser.add_option('-n', '--num-pivots', dest='num_pivots',
                help='number of pivot terms to use in training (default is to use all shared terms)',
                type='int', default=-1)
        parser.add_option('--frequent', dest='frequent',
                action='store_true', default=False,
                help='choose pivot terms by frequency in target corpus'
                     ' (requires that --trg-vocab be supplied with a'
                     ' term-frequency file)')
        (options, args) = parser.parse_args()
        if len(args) != 0 or \
                (not options.src_embf) or \
                ( options.frequent and (not options.trg_vocabf) ) or \
                ( (not options.trg_embf) and (not options.trg_vocabf) ) or \
                (not options.outf):
            parser.print_help()
            exit()
        return options
    options = _cli()

    if options.stopwordf:
        stopwords = readStopwords(options.stopwordf, tolower=True)
    else:
        stopwords = set()

    src_embs = pyemblib.read(options.src_embf, mode=options.src_embf_mode)
    src_vocab = set([k.lower() for k in src_embs.keys()])

    if options.trg_vocabf:
        trg_vocab = readVocab(options.trg_vocabf, tolower=True)
        if not options.frequent:
            trg_vocab = set(trg_vocab.keys())
    else:
        trg_embs = pyemblib.read(options.trg_embf, mode=options.trg_embf_mode)
        trg_vocab = set([k.lower() for k in trg_embs.keys()])

    if options.frequent:
        pivots = frequentPivotTerms(src_vocab, trg_vocab, num_terms=options.num_pivots, stopwords=stopwords)
    else:
        pivots = randomPivotTerms(src_vocab, trg_vocab, num_terms=options.num_pivots, stopwords=stopwords)

    writePivotsFile(pivots, options)
