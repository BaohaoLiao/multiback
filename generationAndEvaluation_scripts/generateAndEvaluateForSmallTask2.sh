#!/bin/bash

DATA=../data/small_task2_bin
DEVTEST_PATH=../data/flores101_dataset/devtest
PRETRAINED_MODEL=../models/small_task2/trans_base_parallel_data/checkpoints/checkpoint_best.pt
SAVE_DIR=gen_devtest
DICT=$DATA/dict.en.txt
LANG_PAIRS=../train_scripts/languagePairsForSmallTask2.txt

LANGS=(
    "en"
    "id"
    "jv"
    "ms"
    "ta"
    "tl"
)
CODES=(
    "eng"
    "ind"
    "jav"
    "msa"
    "tam"
    "tgl"
)

mkdir $SAVE_DIR

for ((i=0; i<${#LANGS[@]}-1; ++i)); do
    for ((j=i+1; j<${#LANGS[@]}; ++j)); do

SRC=${LANGS[i]}
TGT=${LANGS[j]}
echo -e "---------------------------------" >> $SAVE_DIR/log
echo -e "${SRC}-${TGT}" >> $SAVE_DIR/log
echo -e "---------------------------------" >> $SAVE_DIR/log
fairseq-generate \
    $DATA \
    --batch-size 256 \
    --path $PRETRAINED_MODEL \
    --fixed-dictionary $DICT \
    -s $SRC -t $TGT \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --max-len-a 1.5 \
    --max-len-b 20 \
    --min-len 1 \
    --task translation_multi_simple_epoch \
    --lang-pairs $LANG_PAIRS \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn | tee $SAVE_DIR/${SRC}2${TGT}.out
cat $SAVE_DIR/${SRC}2${TGT}.out  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $SAVE_DIR/${SRC}2${TGT}.sys
sacrebleu $DEVTEST_PATH/${CODES[j]}.devtest < $SAVE_DIR/${SRC}2${TGT}.sys --tokenize spm | tee -a $SAVE_DIR/log

SRC=${LANGS[j]}
TGT=${LANGS[i]}
echo -e "---------------------------------" >> $SAVE_DIR/log
echo -e "${SRC}-${TGT}" >> $SAVE_DIR/log
echo -e "---------------------------------" >> $SAVE_DIR/log
fairseq-generate \
    $DATA \
    --batch-size 256 \
    --path $PRETRAINED_MODEL \
    --fixed-dictionary $DICT \
    -s $SRC -t $TGT \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --max-len-a 1.5 \
    --max-len-b 20 \
    --min-len 1 \
    --task translation_multi_simple_epoch \
    --lang-pairs $LANG_PAIRS \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn | tee $SAVE_DIR/${SRC}2${TGT}.gen
cat $SAVE_DIR/${SRC}2${TGT}.gen  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $SAVE_DIR/${SRC}2${TGT}.sys
sacrebleu $DEVTEST_PATH/${CODES[i]}.devtest < $SAVE_DIR/${SRC}2${TGT}.sys --tokenize spm | tee -a $SAVE_DIR/log

    done
done