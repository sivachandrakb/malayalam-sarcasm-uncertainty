import torch
import numpy as np

def mc_dropout_predict(model, input_ids, attention_mask, T=20):
    model.train()  # Enable dropout during inference

    probs_list = []

    for _ in range(T):
        probs, _, _ = model(input_ids, attention_mask)
        probs_list.append(probs.detach().cpu().numpy())

    probs_array = np.array(probs_list)

    mean_probs = probs_array.mean(axis=0)
    epistemic_uncertainty = probs_array.var(axis=0).mean()

    return mean_probs, epistemic_uncertainty
