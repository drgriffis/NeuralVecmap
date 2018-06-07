#!/bin/bash

PYTHON=~/.conda/envs/tensorflow_1.2.0_cpu/bin/python3.6
PYTHONPATH=$(pwd)/lib
DATA=$(pwd)/data

SOURCE=${DATA}/demo_embs_source.txt
TARGET=${DATA}/demo_embs_target.txt
STOPS=${DATA}/english-stopwords.txt
PIVOTS=${DATA}/pivots
MAPPED=${DATA}/demo_embs_mapped.txt

SRC_NBRS=${DATA}/demo_embs_source.nn.txt
TRG_NBRS=${DATA}/demo_embs_target.nn.txt
MAP_NBRS=${DATA}/demo_embs_mapped.nn.txt

if [ ! -d ${DATA} ]; then
    mkdir ${DATA}
fi

# download the demo embedding files to use
# (source has 10k keys, target has 5k; 3k are in common
#  data are subsets of pre-trained FastText embeddings
#  from WikiNews, available at
#    http://fasttext.cc/docs/en/english-vectors)
if [ ! -e ${SOURCE} ]; then
    curl -O http://slate.cse.ohio-state.edu/NeuralVecmap/demo_embs_source.txt
    mv demo_embs_source.txt ${SOURCE}
fi
if [ ! -e ${TARGET} ]; then
    curl -O http://slate.cse.ohio-state.edu/NeuralVecmap/demo_embs_target.txt
    mv demo_embs_target.txt ${TARGET}
fi
# and get pre-extracted NLTK English stopwords, for convenience
if [ ! -e ${STOPS} ]; then
    curl -O http://slate.cse.ohio-state.edu/NeuralVecmap/english-stopwords.txt
    mv english-stopwords.txt ${STOPS}
fi

# get the set of pivot keys
if [ ! -e ${PIVOTS} ]; then
    cd src
    ${PYTHON} -m pivots \
        --src-embeddings ${SOURCE} \
        --trg-embeddings ${TARGET} \
        --output ${PIVOTS} \
        --stopwords ${STOPS}
    cd ../
fi

# map the embeddings with a 5-layer ReLU
if [ ! -e ${MAPPED} ]; then
    cd src
    ${PYTHON} -m learnmap \
        --src-embeddings ${SOURCE} \
        --trg-embeddings ${TARGET} \
        --output ${MAPPED} \
        --pivots ${PIVOTS} \
        --activation relu \
        --num-layers 5 \
        --checkpoint-file ${DATA}/checkpoint
    cd ../
fi

# run nearest neighbor analysis
if [ ! -e ${SRC_NBRS} ]; then
    cd nn-analysis
    ${PYTHON} -m nearest_neighbors \
        ${SOURCE} \
        ${SRC_NBRS} \
        --keys ${PIVOTS}
    cd ../
fi
if [ ! -e ${TRG_NBRS} ]; then
    cd nn-analysis
    ${PYTHON} -m nearest_neighbors \
        ${TARGET} \
        ${TRG_NBRS} \
        --keys ${PIVOTS}
    cd ../
fi
if [ ! -e ${MAP_NBRS} ]; then
    cd nn-analysis
    ${PYTHON} -m nearest_neighbors \
        ${MAPPED} \
        ${MAP_NBRS} \
        --keys ${PIVOTS}
    cd ../
fi
cd nn-analysis
${PYTHON} -m nn_changes \
    ${SRC_NBRS} \
    ${TRG_NBRS} \
    ${MAP_NBRS}
cd ../
