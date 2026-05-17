import torch.nn.functional as F


def discriminator_hinge_loss(real_outputs, fake_outputs):
    loss = 0.0
    for (real_logits, _), (fake_logits, _) in zip(real_outputs, fake_outputs):
        loss = loss + F.relu(1.0 - real_logits).mean()
        loss = loss + F.relu(1.0 + fake_logits).mean()
    return loss


def generator_hinge_loss(fake_outputs):
    loss = 0.0
    for fake_logits, _ in fake_outputs:
        loss = loss + F.relu(1.0 - fake_logits).mean()
    return loss


def feature_matching_loss(real_outputs, fake_outputs):
    loss = 0.0
    n = 0
    for (_, real_feats), (_, fake_feats) in zip(real_outputs, fake_outputs):
        for rf, ff in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n += 1
    if n > 0:
        loss = loss / n
    return loss
