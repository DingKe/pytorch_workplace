int relu_forward_cuda(THCudaTensor *input, THCudaTensor *output);
int relu_backward_cuda(THCudaTensor *grad_output, THCudaTensor *input, THCudaTensor *grad_input);
