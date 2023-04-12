import torch
import torch.nn as nn
import torch.nn.functional as F
'''
还没试
'''
class LR_DAM_Loss(nn.Module):
    '''
    LR_DAM_Loss是一个继承自nn.Module的PyTorch类，它接收一个logits张量和一个targets张量作为输入。logits张量包含每个类别的预测分数，而targets张量包含实际标签。

    在这个类的forward方法中，我们首先计算一些常量，例如one-hot编码的目标，以及真正和假正例的累积TPR和FPR。然后，我们计算损失函数，其中第一个项是对FPR的惩罚，第二项是对分类器在正例上的表现进行的监督。

    最后，我们将所有批次的损失函数平均起来，并将其返回作为输出。

    使用这个类的方法与使用PyTorch中的其他损失函数的方法相同。你可以将其用于训练任何深度神经网络模型，例如用于医学图像分类的卷积神经网络。
    '''
    def __init__(self, alpha=0.2, beta=0.2, gamma=0.2, delta=1.0):
        super(LR_DAM_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, logits, targets):
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        targets = targets.view(-1, 1)

        # 计算一些常量
        targets_one_hot = torch.zeros(batch_size, num_classes).cuda().scatter_(1, targets, 1)
        targets_zero_hot = 1 - targets_one_hot
        ones = torch.ones_like(targets_one_hot)
        zeros = torch.zeros_like(targets_one_hot)

        # 计算TPR和FPR
        scores = F.softmax(logits, dim=1)
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)
        sorted_targets = targets_one_hot[sorted_indices.view(-1), :].view_as(scores)
        cumulative_tp = torch.cumsum(sorted_targets, dim=0)
        cumulative_fp = torch.cumsum(ones - sorted_targets, dim=0)
        cumulative_tpr = cumulative_tp / (targets_one_hot.sum(dim=0) + 1e-6)
        cumulative_fpr = cumulative_fp / (targets_zero_hot.sum(dim=0) + 1e-6)

        # 计算损失函数
        loss = (1 - self.alpha) * (self.beta * (ones - cumulative_tpr).pow(self.gamma) * cumulative_fpr).sum(dim=1) \
            + self.alpha * (self.delta * targets_one_hot * (ones - scores).pow(2) * torch.log(scores)).sum(dim=1)

        return loss.mean()