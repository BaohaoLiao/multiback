#!/bin/bash

ENCODE=../fairseq/scripts/spm_encode.py
MODEL=../pretrained_models/flores101_mm100_615M/sentencepiece.bpe.model
DEV_PATH=../data/flores101_dataset/dev
DEVTEST_PATH=../data/flores101_dataset/devtest
SAVE_DIR=../data/small_task2_bin  # TODO: change to the directory for small task #1
DICT=../pretrained_models/flores101_mm100_615M/dict.txt

mkdir $SAVE_DIR

LANGS=(  #TODO: change to the languages for small task #1
    "en"
    "id"
    "jv"
    "ms"
    "ta"
    "tl"
)
CODES=(  # TODO: change to the languages for small task #1
    "eng"
    "ind"
    "jav"
    "msa"
    "tam"
    "tgl"
)

echo "Processing ..."
mkdir tmp
for ((i=0; i<${#CODES[@]}; ++i)); do
    echo "Preprocess valid.${LANGS[i]} and test.${LANGS[i]}"
    python $ENCODE \
        --model $MODEL \
        --output_format=piece \
        --inputs=${DEV_PATH}/${CODES[i]}.dev \
        --outputs=tmp/valid.${LANGS[i]}

    python $ENCODE \
        --model $MODEL \
        --output_format=piece \
        --inputs=${DEVTEST_PATH}/${CODES[i]}.devtest \
        --outputs=tmp/test.${LANGS[i]}
done

mkdir bpe
for ((i=0; i<${#LANGS[@]}-1; ++i)); do
    for ((j=i+1; j<${#LANGS[@]}; ++j)); do
        for LANG in "${LANGS[i]}" "${LANGS[j]}"; do
            cp tmp/valid.${LANG} bpe/valid.${LANGS[i]}-${LANGS[j]}.${LANG}
            cp tmp/test.${LANG} bpe/test.${LANGS[i]}-${LANGS[j]}.${LANG}
        done
    done
done

echo "Binarizing ..."
for ((i=0; i<${#LANGS[@]}-1; ++i)); do
    for ((j=i+1; j<${#LANGS[@]}; ++j)); do
        echo "Binarize ${LANGS[i]}-${LANGS[j]}"
        fairseq-preprocess \
            --source-lang ${LANGS[i]} --target-lang ${LANGS[j]} \
            --validpref bpe/valid.${LANGS[i]}-${LANGS[j]} \
            --testpref bpe/test.${LANGS[i]}-${LANGS[j]} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir $SAVE_DIR \
            --srcdict $DICT --tgtdict $DICT \
            --workers 40
    done
done

rm -rf tmp bpe