#!/bin/bash/

DATA=../data/small_task2_bin
TOOL=../fairseq/train.py
PRETRAINED_MODEL=../pretrained_models/flores101_mm100_175M/model.pt
SAVE_DIR=../models/small_task2/trans_small_parallel_data

lang_pairs=languagePairsForSmallTask2.txt

python $TOOL \
    $DATA \
    --dataset-impl mmap \
    --sampling-method temperature --sampling-temperature 5.0 \
    --arch transformer_wmt_en_de_big \
    --dropout 0.1 --attention-dropout 0.1 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-normalize-before --decoder-normalize-before \
    --share-all-embeddings \
    --task translation_multi_simple_epoch \
    --encoder-langtok "src" --decoder-langtok \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-08 --adam-betas '(0.9, 0.98)' \
    --fp16 --fp16-init-scale 128  --fp16-scale-tolerance 0.0  --memory-efficient-fp16 \
    --lr-scheduler inverse_sqrt --lr 3e-04 --warmup-init-lr 1e-07 --warmup-updates 2500 \
    --weight-decay 0.0001 \
    --max-tokens 4096 \
    --save-interval 1 --save-interval-updates 3000 --keep-interval-updates 10 \
    --max-epoch 1 \
    --seed 222 \
    --no-progress-bar --log-format simple --log-interval 100 \
    --save-dir $SAVE_DIR/checkpoints  \
    --tensorboard-logdir $SAVE_DIR/curves 2>&1 | tee $SAVE_DIR/train.log