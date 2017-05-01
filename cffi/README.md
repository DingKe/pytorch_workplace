# An example C extension for PyTorch

This example showcases defining ReLU activition with C.

- src: C source code
- functions: the autograd functions
- modules: code of the nn module
- build.py: a small file that compiles your module to be ready to use
- test.py: an example file that loads and uses the extension

```bash
python build.py
python test.py
```
