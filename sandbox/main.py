import math
import time
import numpy as np

def min_max(x, list):
    max_val = max(list)
    min_val = min(list)

    if min_val == max_val:
        return 1

    return (x-min_val)/(max_val-min_val)


def main():

    # arr = np.ones((1000000))

    # # Save timestamp
    # start = time.time()

    # for i in arr:
    #     i = i*2

    # # Save timestamp
    # end = time.time()

    # print(end - start)

    # exit()
    scores = [0.8, 0.8, 0.8, 0.8]
    rep_frac = [0.1, 0.1, 0.1, 0.1]

    # scores = [0.9, 0.5, 0.2, 0.3, 0.68, 0.2, 0.95, 0.8]
    # rep_frac = [0.1, 0.15, 0.2, 0.3, 0.13, 0.14, 0.23, 0.16]

    print(f"scores: {scores}")
    print(f"rep_frac: {rep_frac}")

    rep_frac_sum = sum(rep_frac)

    rep_frac_normalized = []
    for r in rep_frac:
        rep_frac_normalized.append(r/rep_frac_sum)
    print(f"rep_frac_normalized: {rep_frac_normalized}")

    site_score = 0

    for i in range(len(scores)):
        s = scores[i]*(rep_frac[i]/rep_frac_sum)
        site_score += s
        print(f"individually wighted score: {s}")

    print(f"site_score (weighted sum): {site_score}")


if __name__ == "__main__":
    main()