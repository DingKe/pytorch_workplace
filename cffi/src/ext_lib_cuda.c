#include <THC/THC.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int relu_forward_cuda(THCudaTensor *input, THCudaTensor *output)
{
  if (!THCudaTensor_isSameSizeAs(state, input1, input2))
    return 0;
  THCudaTensor_resizeAs(state, output, input1);
  THCudaTensor_clamp(state, output, input, 0, THinf);
  return 1;
}

int relu_backward_cuda(THCudaTensor *grad_output, THCudaTensor *input, THCudaTensor *grad_input)
{
  THCudaTensor_resizeAs(state, grad_input, grad_output);
  THCudaTensor_zero(state, grad_input);

  THCudaLongStorage* size = THCudaFloatTensor_newSizeOf(grad_output);
  THCudaLongStorage *stride = THCudaFloatTensor_newStrideOf(grad_output);
  THCudaByteTensor *mask = THCudaByteTensor_newWithSize(size, stride);

  THCudaFloatTensor_geValue(mask, input, 0);
  THCudaFloatTensor_maskedCopy(grad_input, mask, grad_output);

  return 1;
}
