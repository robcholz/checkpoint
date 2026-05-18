# Fixes

- async: it waits at the forward_begin, causing visible lag at foreground.
- gockpt: single-thread background worker cannot catchup reconstruction, causing lag at the foreground. instead we use a ringbuffer + a threadpool to allow the requests to be buffered and processed.
- gockpt: on commit: `38ddd40332199cafdb52d2807e2c1513de180922`, the overlap_steps was 7, but checkpointing is 2x slower than training, leading to visibl foreground lag.
