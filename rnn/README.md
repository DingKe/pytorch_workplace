Implement RNN (Vanilla, GRU, MGRU, LSTM, LSTMP, LSTMPF) from scratch.

* modules.py - define RNN variants
* test_rnn.py - demo

Gradient is clipped to avoid explosion using pytorch Variable's [register_hook](./modules.py#L9) function.
