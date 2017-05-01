int relu_forward(THFloatTensor *input, THFloatTensor *output);
int relu_backward(THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *grad_input);
