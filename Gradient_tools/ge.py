import torch
# ---- GradientRescale ---- #
class GradientRescaleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output

        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply

# ---- END Gradient Rescale ---- #