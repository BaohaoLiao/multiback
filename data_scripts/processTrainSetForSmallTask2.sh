#!/bin/bash

ENCODE=../fairseq/scripts/spm_encode.py
MODEL=../pretrained_models/flores101_mm100_615M/sentencepiece.bpe.model
DATA_PATH=../data/small_task2_filt_concat  # TODO: change to the raw data directory for small task #1
SAVE_DIR=../data/small_task2_bin  # TODO: change to the directory where you want to save for small task #1
DICT=../pretrained_models/flores101_mm100_615M/dict.txt

LANGS=(  # TODO: change to the languages of small task #1
    "en"
    "id"
    "jv"
    "ms"
    "ta"
    "tl"
)

echo "Processing ..."
mkdir bpe
for ((i=0; i<${#LANGS[@]}-1; ++i)); do
    for ((j=i+1; j<${#LANGS[@]}; ++j)); do
        echo "Preprocess train.${LANGS[i]}-${LANGS[j]}"
        for LANG in "${LANGS[i]}" "${LANGS[j]}"; do
            python $ENCODE \
                --model $MODEL \
                --output_format=piece \
                --inputs=${DATA_PATH}/train.${LANGS[i]}-${LANGS[j]}.${LANG} \
                --outputs=bpe/train.${LANGS[i]}-${LANGS[j]}.${LANG}
        done
    done
done

echo "Binarizing ..."
for ((i=0; i<${#LANGS[@]}-1; ++i)); do
    for ((j=i+1; j<${#LANGS[@]}; ++j)); do
        echo "Binarize train.${LANGS[i]}-${LANGS[j]}"
        fairseq-preprocess \
            --source-lang ${LANGS[i]} --target-lang ${LANGS[j]} \
            --trainpref bpe/train.${LANGS[i]}-${LANGS[j]} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir $SAVE_DIR \
            --srcdict $DICT --tgtdict $DICT \
            --workers 40
    done
done

rm -rf bpe
