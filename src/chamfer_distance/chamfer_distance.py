
import torch

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["../src/chamfer_distance/chamfer_distance.cpp",
                   "../src/chamfer_distance/chamfer_distance.cu"])

# class ChamferDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         batchsize, n, _ = xyz1.size()
#         _, m, _ = xyz2.size()
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#         dist1 = torch.zeros(batchsize, n)
#         dist2 = torch.zeros(batchsize, m)

#         idx1 = torch.zeros(batchsize, n, dtype=torch.int)
#         idx2 = torch.zeros(batchsize, m, dtype=torch.int)

#         if not xyz1.is_cuda:
#             cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
#         else:
#             dist1 = dist1.cuda()
#             dist2 = dist2.cuda()
#             idx1 = idx1.cuda()
#             idx2 = idx2.cuda()
#             cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

#         return dist1, dist2

#     @staticmethod
#     def backward(ctx, graddist1, graddist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

#         graddist1 = graddist1.contiguous()
#         graddist2 = graddist2.contiguous()

#         gradxyz1 = torch.zeros(xyz1.size())
#         gradxyz2 = torch.zeros(xyz2.size())

#         if not graddist1.is_cuda:
#             cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
#         else:
#             gradxyz1 = gradxyz1.cuda()
#             gradxyz2 = gradxyz2.cuda()
#             cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

#         return gradxyz1, gradxyz2

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n, device="cuda")
        dist2 = torch.zeros(batchsize, m, device="cuda")

        idx1 = torch.zeros(batchsize, n, dtype=torch.int, device="cuda")
        idx2 = torch.zeros(batchsize, m, dtype=torch.int, device="cuda")

        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)

class ChamferIndexFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n, device="cuda")
        dist2 = torch.zeros(batchsize, m, device="cuda")

        idx1 = torch.zeros(batchsize, n, dtype=torch.int, device="cuda")
        idx2 = torch.zeros(batchsize, m, dtype=torch.int, device="cuda")

        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        # idx1 = idx1.unsqueeze(-1).repeat([1, 1, 3]).long()
        # idx2 = idx2.unsqueeze(-1).repeat([1, 1, 3]).long()
        # target_1 = xyz2.gather(dim=1, index=idx1)
        # target_2 = xyz1.gather(dim=1, index=idx2)
        # mydist_1 = ((target_1 - xyz1) ** 2).sum(dim=-1)
        # mydist_2 = ((target_2 - xyz2) ** 2).sum(dim=-1)
        # print("err: ", (mydist_1 - dist1).mean(), (mydist_2 - dist2).mean())
        # print(dist1, mydist_1)
        # exit()

        return idx1, idx2


class ChamferIndex(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferIndexFunction.apply(xyz1, xyz2)


if __name__ == '__main__':
    pc1 = torch.randn([2, 600, 3], dtype=torch.float32,
                      requires_grad=True).cuda()
    pc2 = torch.randn([2, 300, 3], dtype=torch.float32,
                      requires_grad=True).cuda()
    chamfer = ChamferIndex()
    print([x.shape for x in chamfer(pc1, pc2)])
