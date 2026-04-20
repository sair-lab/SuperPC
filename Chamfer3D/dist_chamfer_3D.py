from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ])
    #print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    #print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)


def density_aware_chamfer_distance(x_bnc, gt_bnc, alpha=1000.0, n_lambda=1.0, non_reg=False, return_per_batch=False):
    """Compute DCD from Chamfer nearest-neighbor distances and indices.

    Args:
        x_bnc: Predicted points with shape [B, N_x, 3].
        gt_bnc: Ground-truth points with shape [B, N_gt, 3].
        alpha: Exponential scale factor.
        n_lambda: Density reweight exponent.
        non_reg: Use non-regularized frac terms from the original DCD implementation.
        return_per_batch: If True, return shape [B], else return batch mean scalar.
    """
    if x_bnc.ndim != 3 or gt_bnc.ndim != 3 or x_bnc.shape[2] != 3 or gt_bnc.shape[2] != 3:
        raise ValueError(f"Invalid shape for DCD: x={x_bnc.shape}, gt={gt_bnc.shape}")
    if x_bnc.shape[0] != gt_bnc.shape[0]:
        raise ValueError(f"Batch size mismatch for DCD: x={x_bnc.shape[0]}, gt={gt_bnc.shape[0]}")

    x_bnc = x_bnc.float()
    gt_bnc = gt_bnc.float()
    bsz, n_x, _ = x_bnc.shape
    _bsz2, n_gt, _ = gt_bnc.shape

    if non_reg:
        frac_12 = max(1.0, float(n_x) / float(n_gt))
        frac_21 = max(1.0, float(n_gt) / float(n_x))
    else:
        frac_12 = float(n_x) / float(n_gt)
        frac_21 = float(n_gt) / float(n_x)

    if x_bnc.is_cuda and gt_bnc.is_cuda:
        # Match DCD convention in model_utils.calc_cd: chamfer(gt, x).
        dist1, dist2, idx1, idx2 = chamfer_3DFunction.apply(gt_bnc.contiguous(), x_bnc.contiguous())
    else:
        dmat_sq = torch.cdist(gt_bnc, x_bnc, p=2) ** 2
        dist1, idx1 = torch.min(dmat_sq, dim=2)  # gt -> x
        dist2, idx2 = torch.min(dmat_sq, dim=1)  # x -> gt

    exp_dist1 = torch.exp(-dist1 * float(alpha))
    exp_dist2 = torch.exp(-dist2 * float(alpha))

    count1 = torch.zeros((bsz, n_x), dtype=x_bnc.dtype, device=x_bnc.device)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1, dtype=x_bnc.dtype))
    weight1 = count1.gather(1, idx1.long()).detach() ** float(n_lambda)
    weight1 = (weight1 + 1e-6).reciprocal() * float(frac_21)
    loss1 = (1.0 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros((bsz, n_gt), dtype=gt_bnc.dtype, device=gt_bnc.device)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2, dtype=gt_bnc.dtype))
    weight2 = count2.gather(1, idx2.long()).detach() ** float(n_lambda)
    weight2 = (weight2 + 1e-6).reciprocal() * float(frac_12)
    loss2 = (1.0 - exp_dist2 * weight2).mean(dim=1)

    dcd = (loss1 + loss2) / 2.0
    if return_per_batch:
        return dcd
    return dcd.mean()

def hausdorff_distance(X,Y):
    '''
     the HD is from MPU
    Parameters
    ----------
    X
    Y

    Returns
    -------

    '''
    B,N,C = X.shape
    dist1, dist2 ,_ ,_ = chamfer_3DFunction.apply(X, Y)
    h1 = torch.amax(dist1, dim=1).view(B,1)
    h2 = torch.amax(dist2, dim=1).view(B,1)
    hd_loss = torch.cat([h1,h2],dim=-1)
    return hd_loss



if __name__ == '__main__':

    x = torch.zeros(size=(2,1024,3)).cuda()
    y = torch.zeros(size=(2,1024,3)).cuda()

    cd = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cd(x,y)

    print(dist1.shape)
    print(dist2.shape)
