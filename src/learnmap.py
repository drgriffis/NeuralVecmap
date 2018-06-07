'''
Learn a (nonlinear) mapping from one embedding manifold to
another, using a predetermined set of pivot points, and
regularizing with cross-validation.
'''

import codecs
import random
import numpy as np
import pyemblib
import configlogger
import time
import tensorflow as tf
from collections import OrderedDict
from drgriffis.common import log
from model import MapperParams, ManifoldMapper
from pivots import readPivotsFile

def train(model, src_embs, trg_embs, train_keys, dev_keys, batch_size=5):
    
    train_keys = list(train_keys)
    dev_keys = list(dev_keys)
    
    training = True
    batch_start, iter_losses= 0, []
    prev_dev_loss = None
    cur_iter, new_iter = 0, True
    while training:
        
        if new_iter:
            if cur_iter > 0:
                # run on dev set
                dev_loss = evalOnDev(model, src_embs, trg_embs, dev_keys, batch_size=batch_size)
                log.writeln("    Iteration %d -- Dev MSE: %f" % (cur_iter, dev_loss))
                if cur_iter > 1 and dev_loss > prev_dev_loss:
                    training = False
                    log.writeln('    >> Reached dev-based convergence <<')
                else:
                    prev_dev_loss = dev_loss
                    # save checkpoint
                    model.checkpoint(cur_iter)
                
            # set up for next training batch
            random.shuffle(train_keys)
            cur_iter += 1
            batch_start = 0
            iter_losses = []
            new_iter = False

        if training:
            batch_keys = train_keys[batch_start:batch_start+batch_size]
            batch_src = np.array([src_embs[k] for k in batch_keys])
            batch_trg = np.array([trg_embs[k] for k in batch_keys])
            loss = model.train_batch(batch_src, batch_trg)
            iter_losses.append(loss)

            batch_start += batch_size
            if batch_start >= len(train_keys):
                new_iter = True

    model.rollback()


def evalOnDev(model, src_embs, trg_embs, dev_keys, batch_size=5):
    
    batch_start, dev_errors = 0, []
    while batch_start < len(dev_keys):
        batch_keys = dev_keys[batch_start:batch_start+batch_size]
        batch_src = np.array([src_embs[k] for k in batch_keys])
        batch_trg = np.array([trg_embs[k] for k in batch_keys])
        loss = model.eval_batch(batch_src, batch_trg)
        dev_errors.append(loss)
        batch_start += batch_size

    return np.mean(dev_errors)

def crossfoldTrain(src_embs, trg_embs, pivot_keys, nfold, activation, num_layers, batch_size=5, checkpoint_file='checkpoint', random_seed=None): 
    
    project_batch_size = batch_size * 10
    
    pivot_keys = list(pivot_keys)
    if random_seed:
        random.seed(random_seed)
    random.shuffle(pivot_keys)

    fold_size = int(np.ceil(len(pivot_keys) / nfold))

    mapped_embs = {}
    src_keys = list(src_embs.keys())
    for k in src_keys:
        mapped_embs[k] = np.zeros([trg_embs.size])

    session = tf.Session()
    params = MapperParams(
        src_dim=src_embs.size,
        trg_dim=trg_embs.size,
        map_dim=trg_embs.size,
        activation=activation,
        num_layers=num_layers,
        checkpoint_file=checkpoint_file
    )

    for i in range(nfold):
        log.writeln('  Starting fold %d/%d' % (i+1,nfold))
        if random_seed:
            this_random = random_seed + i
        else:
            this_random = None
        model = ManifoldMapper(session, params, random_seed=this_random)

        fold_start, fold_end = (i*fold_size), ((i+1)*fold_size)
        train_keys = pivot_keys[:fold_start]
        dev_keys = pivot_keys[fold_start:fold_end]
        train_keys.extend(pivot_keys[fold_end:])

        train(model, src_embs, trg_embs, train_keys, dev_keys, batch_size=batch_size)

        # get projections from this fold
        log.writeln('  Getting trained projections for fold %d' % (i+1))
        log.track(message='    >> Projected {0}/%d keys' % len(src_keys), writeInterval=10000)
        batch_start = 0
        while batch_start < len(src_keys):
            batch_keys = src_keys[batch_start:batch_start+project_batch_size]
            batch_src = np.array([src_embs[k] for k in batch_keys])
            batch_mapped = model.project_batch(batch_src)

            for i in range(batch_mapped.shape[0]):
                key = batch_keys[i]
                mapped_embs[key] += batch_mapped[i]
                log.tick()

            batch_start += project_batch_size
        log.flushTracker()

    # mean projections
    for k in src_keys:
        mapped_embs[k] /= nfold

    # get final MSE over full pivot set
    final_errors = []
    for k in pivot_keys:
        diff = mapped_embs[k] - trg_embs[k]
        final_errors.append(np.sum(diff ** 2) / 2)
    log.writeln('\nPivot error in final projections: %f' % np.mean(final_errors))

    return mapped_embs


if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('--src-embeddings', dest='src_embf',
                help='source embeddings file')
        parser.add_option('--src-embeddings-mode', dest='src_embf_mode',
                help='source embeddings file mode (default: %default)',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text)
        parser.add_option('--trg-embeddings', dest='trg_embf',
                help='target embeddings file')
        parser.add_option('--trg-embeddings-mode', dest='trg_embf_mode',
                help='target embeddings file mode (default: %default)',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text)
        parser.add_option('--pivots', dest='pivotf',
                help='pivot terms file (REQUIRED)')
        parser.add_option('--output', dest='outf',
                help='file to write mapped embeddings to')
        parser.add_option('--output-mode', dest='out_embf_mode',
                help='output embeddings file mode (default: %default)',
                type='choice', choices=[pyemblib.Mode.Binary, pyemblib.Mode.Text],
                default=pyemblib.Mode.Text)
        parser.add_option('-n', '--num-pivots', dest='num_pivots',
                help='number of pivot terms to use in training (default: %default)',
                type='int', default=10000)
        parser.add_option('--activation', dest='activation',
                help='nonlinearity for projection model (options: tanh, relu; default: %default)',
                type='choice', choices=['tanh','relu'], default='relu')
        parser.add_option('--num-layers', dest='num_layers',
                help='number of hidden layers in projection model (default: %default)',
                type='int', default=1)
        parser.add_option('--checkpoint-file', dest='checkpointf',
                help='base filename for checkpoint files (used in calculating final projection; default: %default)',
                default='checkpoint')
        parser.add_option('--random-seed', dest='random_seed',
                help='value to seed the random initialization with (defaults to current epoch time)',
                default=-1)
        parser.add_option('-k', '--num-folds', dest='num_folds',
                help='number of folds to divide pivot terms into for cross-validation training (default: %default)',
                type='int', default=10)
        parser.add_option('--batch-size', dest='batch_size',
                help='batch size for model training (default: %default)',
                type='int', default=5)
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 0 or (not options.src_embf) or (not options.trg_embf) or (not options.outf) or (not options.pivotf):
            parser.print_help()
            exit()
        return options
    options = _cli()
    log.start(logfile=options.logfile, stdout_also=True)

    # set the random seed here if necessary
    if options.random_seed <= 0:
        options.random_seed = int(time.time())

    t_sub = log.startTimer('Reading source embeddings from %s...' % options.src_embf, newline=False)
    src_embs = pyemblib.read(options.src_embf, mode=options.src_embf_mode, lower_keys=True)
    log.stopTimer(t_sub, message='Read %d embeddings in {0:.2f}s' % len(src_embs))

    t_sub = log.startTimer('Reading target embeddings from %s...' % options.trg_embf, newline=False)
    trg_embs = pyemblib.read(options.trg_embf, mode=options.trg_embf_mode, lower_keys=True)
    log.stopTimer(t_sub, message='Read %d embeddings in {0:.2f}s' % len(trg_embs))

    pivots = readPivotsFile(options.pivotf, tolower=True)
    log.writeln('Loaded %d pivot terms.' % len(pivots))

    # double check that pivots are present in both embedding files
    validated_pivots = set()
    for pivot in pivots:
        if (not pivot in src_embs) or (not pivot in trg_embs):
            log.writeln('[WARNING] Pivot term "%s" not found in at least one embedding set' % pivot)
        else:
            validated_pivots.add(pivot)

    # write the experimental configuration
    configlogger.writeConfig(
        '%s.config' % options.checkpointf,
        title='DNN embedding mapping experiment',
        settings=[
            ('Source embeddings', options.src_embf),
            ('Source embedding dimension', src_embs.size),
            ('Target embeddings', options.trg_embf),
            ('Target embedding dimension', trg_embs.size),
            ('Output file', options.outf),
            ('Pivot file', options.pivotf),
            ('Number of validated pivots', len(validated_pivots)),
            ('Checkpoint file', options.checkpointf),
            ('Model settings', OrderedDict([
                ('Random seed', options.random_seed),
                ('Number of layers', options.num_layers),
                ('Activation', options.activation),
                ('Number of folds', options.num_folds),
                ('Batch size', options.batch_size),
            ]))
        ]
    )

    log.writeln('Training manifold mapper...')
    mapped_embs = crossfoldTrain(src_embs, trg_embs, validated_pivots, options.num_folds, options.activation, options.num_layers, batch_size=options.batch_size, checkpoint_file=options.checkpointf, random_seed=options.random_seed)

    if options.outf:
        log.writeln('Writing mapped embeddings to %s' % options.outf)
        pyemblib.write(mapped_embs, options.outf, verbose=True, mode=options.out_embf_mode)
