def knap_violation(ind, items_dict, capacity):
    # items_dict[i] -> (value, weight); feasible iff total_w - capacity <= 0
    total_w = sum(int(ind[i]) * items_dict[i][1] for i in range(len(ind)))
    return float(total_w - capacity)