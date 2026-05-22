# Fixes

- async: it waits at the forward_begin, causing visible lag at foreground.
- gockpt: single-thread background worker cannot catchup reconstruction, causing lag at the foreground. instead we use a ringbuffer + a threadpool to allow the requests to be buffered and processed.
- gockpt: on commit: `38ddd40332199cafdb52d2807e2c1513de180922`, the overlap_steps was 7, but checkpointing is 2x slower than training, leading to visibl foreground lag.

## Problem

 Yes, your assumption is basically right.

  1. Why is update so small?
  In this benchmark, “update” is only this part of the loop: finetune.py:480

  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  scheduler.step()

  It does not include backward. For Qwen3-0.6B, forward/backward does transformer
  matmuls over seq-len 512, and gradient checkpointing also makes backward recompute
  activations. Update is just AdamW elementwise tensor updates on GPU, plus
  zero_grad(set_to_none=True), which is cheap. So your measured update avg is only
  about:

  gockpt_o raw update avg: ~0.0042s
  gockpt_o raw F+B avg:   ~0.2806s

  That is why the update window is too small to hide much transfer.

  2. Why does gradient transfer take long?
  Because GoCkpt does not transfer only one full gradient per checkpoint. With
  overlap_steps = K, early M+O partitions need gradients for multiple later steps.

  Approx formula:

  single full gradient size = P

  GoCkpt gradient traffic per checkpoint
  = P * (K + 1) / 2

  For your K = 7:

  gradient traffic ~= 4P

  Now compare with model+optimizer state:

  P = model parameter bytes

  M only                ~= 1P
  gradient once         ~= 1P
  M+O with bf16 moments ~= 3P   # param + exp_avg + exp_avg_sq
  M+O with fp32 moments ~= 5P
  GoCkpt gradients, K=7 ~= 4P

  For Qwen3-0.6B with bf16/fp16 params:

  P ~= 0.6B * 2 bytes ~= 1.2 GB

  single gradient       ~= 1.2 GB
  M+O bf16 moments      ~= 3.6 GB
  M+O fp32 moments      ~= 6.0 GB
  GoCkpt gradients K=7  ~= 4.8 GB

  So yes: gradient traffic being comparable to M+O traffic is normal. With bf16
  optimizer moments, total gradient traffic can even be larger than M+O: 4P / 3P =
  1.33x.

  The extra reason it hurts is timing: M+O can start in forward_begin and overlap with
  F+B. Gradients only exist after backward, then GoCkpt-O submits the copy in
  update_begin and must finish it by the next backward_begin: src/gockpt_o_hook.py:91.
  Since update is tiny, the remaining gradient copy leaks into foreground.

  So the clean conclusion is:

  M+O is mostly hideable because F+B is large. Gradient transfer is harder to hide
  because it appears late and its total traffic is about 4 full-gradient copies per
  checkpoint for overlap_steps=7. That is why it worsens GoCkpt/GoCkpt-O versus Async-
  O.
