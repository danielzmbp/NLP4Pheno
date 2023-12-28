task=NCBI-disease_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.jsonl --validation_file $datadir/dev.jsonl --test_file $datadir/test.jsonl \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 10 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &