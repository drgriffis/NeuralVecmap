'''
'''

import codecs
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import pyemblib
from drgriffis.common import log, util

class NearestNeighbors:
    
    def __init__(self, session, embed_array):
        # unit norm the embedding array
        embed_array = np.array([
            vec / np.linalg.norm(vec)
                for vec in embed_array
        ])

        self._session = session
        self._prints = []
        
        self._dim = embed_array.shape[1]

        self._build(embed_array.shape)

        self._session.run(tf.global_variables_initializer())

        # fill the (static) embedding matrix
        self._session.run(self._embed_matrix.assign(self._embed_ph), feed_dict={self._embed_ph: embed_array})

    def _build(self, emb_shape):
        self._sample_indices = tf.placeholder(
            shape=[None,],
            dtype=tf.int32
        )
        self._embed_ph = tf.placeholder(
            shape=emb_shape,
            dtype=tf.float32
        )
        self._embed_matrix = tf.Variable(
            tf.constant(0.0, shape=emb_shape),
            trainable=False
        )
        self._sample_points = tf.gather(
            self._embed_matrix,
            self._sample_indices
        )

        self._sample_distances = self._distance(self._sample_points, self._embed_matrix)

    def _distance(self, a, b):
        normed_a = a
        normed_b = b   # already unit-normed
        # get full pairwise distance matrix
        pairwise_distance = 1 - tf.matmul(normed_a, normed_b, transpose_b=True)
        return pairwise_distance

    def _print(self, *nodes):
        for n in nodes:
            if type(n) is tuple and len(n) == 2:
                self._prints.append(tf.Print(0, [n[0]], message=n[1], summarize=100))
            else:
                self._prints.append(tf.Print(0, [n], summarize=100))

    def _exec(self, nodes, feed_dict=None):
        all_nodes = [p for p in self._prints]
        all_nodes.extend(nodes)
        outputs = self._session.run(all_nodes, feed_dict=feed_dict)
        return outputs[len(self._prints):]

    def nearestNeighbors(self, batch_indices, top_k=None, no_self=True):
        (pairwise_distances,) = self._exec([
                self._sample_distances
            ],
            feed_dict = {
                self._sample_indices: batch_indices
            }
        )

        nearest_neighbors = []
        for i in range(len(batch_indices)):
            distance_vector = pairwise_distances[i]
            sorted_neighbors = np.argsort(distance_vector)
            # if skipping the query, remove it from the neighbor list
            # (should be in the 0th position; if it's not, just move on)
            if no_self: 
                if sorted_neighbors[0] == batch_indices[i]: sorted_neighbors = sorted_neighbors[1:]
            if top_k is None: nearest_neighbors.append(sorted_neighbors)
            else: nearest_neighbors.append(sorted_neighbors[:top_k])
        return nearest_neighbors

class _SIGNALS:
    HALT=-1

def _threadedNearestNeighbors(thread_ixes, batch_size, top_k, emb_list, nn_q):
    sess = tf.Session()
    nn_calculator = NearestNeighbors(sess, emb_list)

    batch_start = 0
    while batch_start < len(thread_ixes):
        batch_ixes = thread_ixes[batch_start:batch_start+batch_size]
        nearest_neighbors = nn_calculator.nearestNeighbors(batch_ixes, top_k=top_k)
        nn_q.put((batch_ixes, nearest_neighbors))
        batch_start += batch_size

def _collate(keys, n_batches, nn_q, outf):
    with codecs.open(outf, 'w', 'utf-8') as stream:
        result = nn_q.get()
        log.track(message='  >> Processed {0}/%d batches' % n_batches, writeInterval=5)
        while result != _SIGNALS.HALT:
            (batch_ixes, batch_nns) = result
            for i in range(len(batch_ixes)):
                stream.write('%s\t%s\n' % (
                    keys[batch_ixes[i]], 
                    ','.join([keys[ix] for ix in batch_nns[i]])
                ))
            log.tick()
            result = nn_q.get()
        log.flushTracker()

def calculateNearestNeighbors(embeds, outf, top_k=100, batch_size=100, threads=1):
    log.writeln('Calculating nearest neighbors')

    keys = tuple(embeds.keys())
    emb_list = [embeds[k] for k in keys]

    all_ixes = range(len(keys))
    thread_chunks = util.prepareForParallel(all_ixes, threads, data_only=True)

    nn_q = mp.Queue()

    calc_threads = [
        mp.Process(target=_threadedNearestNeighbors, args=(thread_chunks[i], batch_size, top_k, emb_list, nn_q))
            for i in range(threads)
    ]
    collator = mp.Process(target=_collate, args=(keys, (len(keys)//batch_size)+1, nn_q, outf))

    collator.start()
    util.parallelExecute(calc_threads)
    nn_q.put(_SIGNALS.HALT)
    collator.join()

def readNearestNeighbors(f):
    # point neighbors are not written in index order, so pull with indices first
    neighbors = {}
    with open(f, 'r') as stream:
        for line in stream:
            if line.strip() == '': continue
            (src, nbrs) = [s.strip() for s in line.split('\t')]
            src = int(src)
            nbrs = [int(nbr) for nbr in nbrs.split(',')]
            neighbors[src] = nbrs
    # collapse into indexed numpy array
    neighbor_arr = np.array([
        neighbors[i] for i in range(len(neighbors))
    ])
    return neighbor_arr

def readKeys(f):
    keys = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            keys.add(line.strip())
    return keys

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBEDS OUTF',
                description='Writes nearest neighbors for every point in EMBEDS to OUTF')
        parser.add_option('--mode', dest='mode',
                help='file format of EMBEDS: choices [%s, %s] (default: %s)' % (pyemblib.Mode.Binary, pyemblib.Mode.Text, "%default"),
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text], default=pyemblib.Mode.Text)
        parser.add_option('-k', '--top-k', dest='top_k',
                help='number of nearest neighbors to get (default: %default)',
                type='int', default=10)
        parser.add_option('-t', '--threads', dest='threads',
                help='number of threads to use for parallel calculation (default: %default)',
                type='int', default=1)
        parser.add_option('--batch-size', dest='batch_size',
                help='number of samples to process in each batch (default: %default)',
                type='int', default=25)
        parser.add_option('--keys', dest='keysf',
                help='file listing keys to restrict NN analysis to')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 2:
            parser.print_help()
            exit()
        embf, outf = args
        return embf, options.mode, options.keysf, outf, options.top_k, options.batch_size, options.threads, options.logfile
    embf, embf_mode, keysf, outf, top_k, batch_size, threads, logfile = _cli()

    if keysf:
        keys = readKeys(keysf)
        print("Read %d keys to restrict to" % len(keys))
    else:
        keys = None

    t = log.startTimer('Reading embeddings...', newline=False)
    embeds = pyemblib.read(embf, mode=embf_mode, filter_to=keys, lower_keys=True)
    log.stopTimer(t, message='Done! Read %d embeddings ({0:.2f}s)' % len(embeds))

    nearest_neighbors = calculateNearestNeighbors(embeds, outf, top_k=top_k, batch_size=batch_size, threads=threads)
    log.writeln('Wrote nearest neighbors to %s.' % outf)
