'''
Check how often the nearest neighbor of a word changed after domain adaptation
'''

import codecs
import os

def readNearestNeighbors(f):
    neighbors = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (word, nbrs) = [s.strip() for s in line.split('\t')]
            nbrs = [s.strip() for s in nbrs.split(',')]
            neighbors[word] = nbrs
    return neighbors

def countNNChanges(before, after, reference):
    changed, total = 0, 0
    changed_to_reference = 0

    for key in before.keys():
        before_nbrs = before[key]
        after_nbrs = after[key]
        ref_nbrs = reference[key]

        if before_nbrs[0] != after_nbrs[0]:
            changed += 1
            if after_nbrs[0] == ref_nbrs[0]:
                changed_to_reference += 1
        total += 1

    return total, changed, changed_to_reference

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog SRC_NBRF TRG_NBRF MAP_NBRF')
        (options, args) = parser.parse_args()
        if len(args) != 3:
            parser.print_help()
            exit()
        return args
    (src_nbrf, trg_nbrf, map_nbrf) = _cli()
    src_nbrs = readNearestNeighbors(src_nbrf)
    trg_nbrs = readNearestNeighbors(trg_nbrf)
    map_nbrs = readNearestNeighbors(map_nbrf)
    (total, changed, changed_to_reference) = countNNChanges(src_nbrs, map_nbrs, trg_nbrs)
    print("# of words whose nearest neighbor changed after mapping: %d/%d" % (changed, total))
    print("# of words whose nearest neighbor switched to the target's neighbor: %d/%d" % (changed_to_reference, changed))
