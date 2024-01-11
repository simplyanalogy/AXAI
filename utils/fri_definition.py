import numpy as np

from .data_generation import *


# Dag is never empty because a is always distinct from b
def Dag(a, b):  # disagreement set between Boolean vectors of same dimension
    disagreement_indices = [i for i, (v1, v2) in enumerate(zip(a, b)) if v1 != v2]
    return disagreement_indices


def Ag(a, b):
    agreement_indices = [i for i, (v1, v2) in enumerate(zip(a, b)) if v1 == v2]
    return agreement_indices


# Hamming distance between Boolean vectors of same dimension - whatever the dimension
def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def Dif_no_weight(dim, i, set_of_pairs, max_distance):
    ag_att = 0  # Number of pairs such that: Dag={attribute} |Dif(i)|
    m_att = 0  # Number of pairs such that: Dag={attribute} with different label |Dif(i) - Dif_Eq(i)|
    for a, b in set_of_pairs:
        dag_list = Dag(a[:dim], b[:dim])
        if i in dag_list and len(dag_list) <= max_distance:
            ag_att += 1
            if a[dim] != b[dim]:
                m_att += 1
    return ag_att, m_att  # FRI = m_att / ag_att


def fri(dimension, i, set_of_pairs, max_distance):  # FRI(i) for feature i
    ag_att, m_att = Dif_no_weight(dimension, i, set_of_pairs, max_distance)
    if ag_att == 0:  # all agree on i
        return 0.0
    return m_att / ag_att


def get_fri_scores(
    dimension, sample_set, max_distance
):  # return a list of score per feature
    set_of_pairs = all_pairs(sample_set)
    fri_scores = []
    for i in range(dimension):
        ratio = fri(dimension, i, set_of_pairs, max_distance)
        fri_scores.append(ratio)
    return fri_scores  # , list_of_numbers_a1


# PRE_PROCESSING
def estimate_fri_score_on_data(data, max_distance):  # data is a dataframe
    dimension = data.shape[1] - 1
    mean_fri_scores = [0] * dimension
    fri_scores = get_fri_scores(dimension, data, max_distance)
    for j in range(dimension):
        mean_fri_scores[j] += fri_scores[j]  # /(Z_fri_4)
        mean_fri_scores = [round(a, 3) for a in mean_fri_scores]
    return mean_fri_scores
