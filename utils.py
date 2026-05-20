import torch


def evidential_loss(y, alpha):

    S = torch.sum(alpha, dim=1, keepdim=True)

    probs = alpha / S

    error = torch.sum(
        (y - probs) ** 2,
        dim=1,
        keepdim=True
    )

    variance = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)),
        dim=1,
        keepdim=True
    )

    loss = error + variance

    return loss.mean()


def predictive_entropy(probs):

    return -torch.sum(
        probs * torch.log(probs + 1e-10),
        dim=-1
    )


def mutual_information(mc_probs):

    mean_probs = mc_probs.mean(dim=0)

    entropy_mean = predictive_entropy(mean_probs)

    entropy_each = predictive_entropy(mc_probs)

    mean_entropy = entropy_each.mean(dim=0)

    return entropy_mean - mean_entropy
