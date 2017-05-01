#include <TH/TH.h>

int relu_forward(THFloatTensor *input, THFloatTensor *output)
{
  THFloatTensor_resizeAs(output, input);
  THFloatTensor_clamp(output, input, 0, INFINITY);
  return 1;
}

int relu_backward(THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_zero(grad_input);

  THLongStorage* size = THFloatTensor_newSizeOf(grad_output);
  THLongStorage *stride = THFloatTensor_newStrideOf(grad_output);
  THByteTensor *mask = THByteTensor_newWithSize(size, stride);

  THFloatTensor_geValue(mask, input, 0);
  THFloatTensor_maskedCopy(grad_input, mask, grad_output);
  return 1;
}
