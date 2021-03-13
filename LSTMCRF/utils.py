# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(x):
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()