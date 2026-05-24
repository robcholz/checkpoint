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



  The real solution is to change the inequality, not just tune chunk size.

  Right now GoCkpt-O needs:

  gradient transfer <= update + next forward

  Your sweep shows gradient transfer is still about 0.14s per gradient-transfer step,
  while the hide window is around 0.10s. Chunk tuning helps throughput a bit, but does
  not solve the mismatch. 64 MiB is the best measured chunk size, so I fixed the stale
  README note to match that; train.sh was already using 64.

  The actual fixes are, in priority order:

  1. Extend the gradient hide window
     Change GoCkpt-O so it does not force gradient transfer to finish at the next
     backward_begin. Keep old gradient tensors alive and allow a bounded lag of 2-3
     steps, then finish transfers opportunistically. This is the most direct code fix.
     Tradeoff: more GPU memory because pending gradients stay alive longer.
  2. Reduce gradient transfer latency
     Preallocate/reuse pinned CPU gradient buffers instead of allocating pinned memory
     every transfer, and ideally bucket/flatten gradient copies. Current code copies
     many tensors with repeated pinned allocations; that hurts effective bandwidth.
  3. Fix system placement
     Run with GPU-local NUMA binding/CPU affinity. The paper explicitly used CPU/GPU
     affinity and strong NUMA placement. If pinned host memory is allocated on the
     wrong NUMA node, GPU-to-CPU bandwidth can be much worse.
  4. Increase compute window only for reproduction
     Longer sequence length or heavier forward makes next forward longer, so gradient
     transfer hides better. This may match the paper more closely, but it is not a
     real systems fix because it changes the workload.

  So the code-level solution I would target is: bounded delayed gradient completion
  for GoCkpt-O plus pinned buffer reuse. Chunk size 64 is just the best current
  setting, not the real fix.
