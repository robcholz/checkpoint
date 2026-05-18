# Fixes

- async: it waits at the forward_begin, causing visible lag at foreground.
- gockpt: single-thread background worker cannot catchup reconstruction, causing lag at the foreground. instead we use a ringbuffer + a threadpool to allow the requests to be buffered and processed.
