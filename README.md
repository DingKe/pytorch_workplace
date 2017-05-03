Learn pytorch with examples.

## [basic](./basic)
Toy example shows how to write customized Function and Module

## [rnn](./rnn)
Implementation RNN (Vanilla, GRU, LSTM, LSTMP) from scratch.
Gradient is clipped to avoid explosion, using pytorch Variable's register_hook function.

## [binary](./binary)
BinaryNet with pytorch.
Manipulate learning by 1) modifying optimizer ([mlp.py](./binary/adam.py#L72)) or 2) using param_groups
([cnn.py](binary/cnn.py#L70)).

## [cffi](./cffi)
Extend pytorch with cffi.

