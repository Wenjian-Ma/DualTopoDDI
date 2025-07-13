import torch
from torch import nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature

    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2, p_loss, n_loss




def InfoNCE(view1_list, view2_list, temperature=0.2, b_cos: bool = True,device = None):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """

    total_score = torch.Tensor().to(device)

    for idx in range(len(view1_list)):
        view1 = view1_list[idx]
        view2 = view2_list[idx]



        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        # pos_score = (view1 @ view2.T) / temperature
        # score = torch.diag(F.logsigmoid(pos_score))

        score = F.logsigmoid(torch.sum((view1 * view2) / temperature,dim=1))

        total_score = torch.cat((total_score, score), 0)


    return -total_score.mean()



