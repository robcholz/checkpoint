# Checkpoint

## Verify

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 200 \
  --save-steps 20 \
  --gradient-checkpointing
```

## Finetuning

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 1000 \
  --save-steps 20 \
  --gradient-checkpointing
```

