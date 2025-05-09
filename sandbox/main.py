import math

def min_max(x, list):
    max_val = max(list)
    min_val = min(list)

    if min_val == max_val:
        return 1

    return (x-min_val)/(max_val-min_val)


def main():

    w = [0.1, 0.15, 0.2, 0.3, 0.33, 0.4, 0.53, 0.6]

    # w = [0.6]
    # w = [0.6, 0.9]
    # w = [0.5, 0.5]
    # w = [0.75, 0.75]
    # w = [0.1, 0.5]

    # Each step, add to w. Keep tabs on sum, min and max. 
    # Loop through w, get normalized list. Keep tabs on the sum of this list.
    # Loop through normalized list, get weights by dividing by sum of normals. 

    w_normalized = []

    print(f"list: {w}")
    print(f"list sum: {sum(w)}")

    for i in w:
        w_normalized.append(min_max(i, w))

    su = sum(w_normalized)
    print(f"normalzed list: {w_normalized}")
    print(f"normalzed sum: {sum(w_normalized)}")

    w_normalized_div = []
    
    for i in w_normalized:
        w_normalized_div.append(i/su)

    print(f"normalzed div list: {w_normalized_div}")
    print(f"normalzed div sum: {sum(w_normalized_div)}")

if __name__ == "__main__":
    main()