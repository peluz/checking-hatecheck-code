# Based on the MIT-licensed implementation (https://gist.github.com/dustalov/e6c3b9d3b5b83c81ecd92976e0281d6c) of the sigf toolkit for randomization tests:
# https://nlpado.de/~sebastian/software/sigf.shtml

import random
import sys
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm


def randomized_test(model1, model2, labels, trials, seed=42):
    score1 = f1_score(labels, model1, average="macro")
    score2 = f1_score(labels, model2, average="macro")
    print('# score(model1) = %f' % score1)
    print('# score(model2) = %f' % score2)

    diff = abs(score1 - score2)
    print('# abs(diff) = %f' % diff)

    uncommon = [i for i in range(len(model1)) if model1[i] != model2[i]]

    better = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        model1_local, model2_local = list(model1), list(model2)

        for i in uncommon:
            if getrandbits_func(1) == 1:
                model1_local[i], model2_local[i] = model2[i], model1[i]

        assert len(model1_local) == len(model2_local) == len(model1) == len(model2)

        diff_local = abs(f1_score(labels, model1_local, average="macro") - f1_score(labels, model2_local, average="macro"))

        if diff_local >= diff:
            better += 1

    p = (better + 1) / (trials + 1)
    print(f"p_value: {p}, successes: {better}")
    return p