from .fri_definition import *
from .test_functions_definition import *


# a is always a vector + its label
def atomic_analogy(a, b, c, d):
    return (a == b and c == d) or (a == c and b == d)  # gggg ghgh gghh


def vector_analogy(a, b, c, d, m):  # we watch only the relevant features
    res = True
    l = len(m)
    for i in range(l):
        if m[i] == 1:
            res = res and atomic_analogy(a[i], b[i], c[i], d[i])
    return res


def hamming_on_mask(a, b, m):  # compute hamming distance on relevant features
    hamming = 0
    l = len(m)
    for i in range(l):
        if m[i] == 1:
            if a[i] != b[i]:
                hamming += 1
    return hamming


# create a mask = Mask array indicating relevant (1) and irrelevant (0) features. - len(mask)=dimension
def get_indices_of_k_largest_elements(lst, k):
    # Enumerate the list to keep track of indices
    indexed_list = list(enumerate(lst))
    # Sort the list of tuples based on the second element (the actual number)
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # Extract the indices of the k greatest elements
    result_indices = [index for index, _ in sorted_list[:k]]
    return result_indices


def create_mask(list_of_scores, threshold):
    dimension = len(list_of_scores)
    m = [1] * dimension  # Initialize mask with all features considered relevant (1)
    # Update mask based on the threshold
    for i in range(dimension):
        if list_of_scores[i] < threshold:
            m[i] = 0
    return m


# k-nn
def order_neighbors(X, a, k, m):
    distances = []
    for i in range(X.shape[0]):
        x = X[i]
        ham = hamming_on_mask(
            x, a, m
        )  # compute hamming on relevant features vectors - not including class
        if ham != 0:
            distances.append((x, ham))
    distances.sort(key=lambda l: l[1])
    nn_neighbors = [d[0] for d in distances]  # exclude the first element (a itself)
    nn_distances = [d[1] for d in distances]
    return nn_neighbors, nn_distances


# get both first with same class and different class
def get_nln_and_nun(a, list_of_nn):
    cl_a = a[-1]
    nln = None
    nun = None
    for x in list_of_nn:
        # print(x)
        cl_x = x[-1]
        if cl_x == cl_a and nln is None:
            nln = x
        elif cl_x != cl_a and nun is None:
            nun = x
        if nln is not None and nun is not None:
            break
    return np.array(nln), np.array(nun)

def explain_d_from_data(data, d, sample_size, m, set_of_pairs):
    # Start working to explain D
    nn_neighbors, nn_distances = order_neighbors(data, d, sample_size, m)
    nln, nun = get_nln_and_nun(d, nn_neighbors)
    c = nun
    # Working with pairs and profiles
    alpha, beta = 0, 0
    for a, b in set_of_pairs:  # Pairs with the same profile on relevant attributes
        if vector_analogy(a, b, c, d, m):
            beta += 1
            if a[-1] != b[-1]:  # cl(a) != cl(b)
                alpha += 1
    
    actual_relevant_attributes = [z + 1 for z in Dag(c, d)]
    return c, alpha, beta, actual_relevant_attributes


def explanation_loop(data, max_distance):
    # GETTING SCORE ESTIMATION FROM DATASET THEN RANK THEN KEEP ONLY THE MORE RELEVANT ONES USING THRESHOLD
    mean_fri_scores = estimate_fri_score_on_data(data, max_distance)
    dimension = data.shape[1] - 1
    threshold = sum(mean_fri_scores) / dimension # act as threshold
    m = create_mask(mean_fri_scores, threshold)
    # PICK A RANDOM ELEMENT D TO BE EXPLAINED
    d = pick_a_random_element(data)
    # EXPLAIN D
    set_of_pairs = all_pairs(data)
    sample_size=data.shape[0]
    c, alpha, beta, actual_relevant_attributes = explain_d_from_data(
        data, d, sample_size, m, set_of_pairs
    )
    # GETTING THE ESTIMATED RELEVANT ATTRIBUTES FOR THE PAIR (c,d)
    global_list_of_relevant_attributes = [i + 1 for i in range(dimension) if m[i] == 1]
    #set1 = set(global_list_of_relevant_attributes)
    #set2 = set(actual_relevant_attributes)
    #actual_relevant_attributes = list(set1.intersection(set2))
    actual_relevant_attributes = list(set(global_list_of_relevant_attributes) & set(actual_relevant_attributes)) 
    return len(set_of_pairs), global_list_of_relevant_attributes, d, c, alpha, beta, actual_relevant_attributes


