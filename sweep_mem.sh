for seqlen in {256..2048..256}
do
falcontune finetune \
    --model=falcon-7b \
    --weights=/datadrive/models/falcon-7b/ \
    --dataset=/datadrive/dataset/shareGPT/dummy.json \
    --data_type=shareGPT \
    --lora_out_dir=/datadrive/falcontune-checkpoints/falcon-7b-shareGPT/ \
    --mbatch_size=4 \
    --batch_size=128 \
    --epochs=3 \
    --lr=3e-4 \
    --cutoff_len=$seqlen \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --warmup_steps=5 \
    --save_steps=500 \
    --save_total_limit=1000 \
    --logging_steps=20 \
    --target_modules='["query_key_value"]' \
    --val_set_size=0 \
    --gradient_checkpointing
done