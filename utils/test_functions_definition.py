import numpy as np


############# UTILITIES
def noteq(x, y):
    if x != y:
        return 1
    return 0


############# FUNCTIONS - ANY DIMENSION >= 20 - arr is an array of values


def g1(arr):
    return noteq(arr[0], arr[1])


def g2(arr):
    if (noteq(arr[0], 0) == 1 or noteq(arr[1], 0) == 1 or noteq(arr[2], 0)) == 1 and (
        noteq(arr[3], 0) == 1 or (noteq(arr[4], 0) == 1 or noteq(arr[5], 0) == 1)
    ):
        return 1
    return 0


def g3(arr):  # XOR combination
    return noteq(noteq(arr[0], arr[1]), arr[2])


def g4(arr):  # using all attributes - all of them have the same FRI
    if sum(arr) == 3:
        return 1
    return 0


def g5(arr):  # using arr[9] to arr[18] - 10 middle features - easy to verify
    return noteq(
        noteq(
            noteq(noteq(arr[9], arr[10]), noteq(arr[11], arr[12])),
            noteq(arr[13], arr[14]),
        ),
        noteq(noteq(arr[15], arr[16]), noteq(arr[17], arr[18])),
    )


def g_example(arr):
    if (arr[0] == 1) and (arr[1] == 1):  # time and group
        return 1
    return 0


def get_k_best_features(l, k, real_list):
    sorted_list = sorted(range(len(l)), key=lambda i: l[i], reverse=True)
    # sorted_list will contain the indices of my_list sorted in descending order based on their values
    sorted_my_list = [l[i] for i in sorted_list]
    # sorted_my_list will contain the values of my_list sorted in descending order
    best_k_features = []
    number_of_correct = 0
    for i in range(k):
        best_k_features.append(
            "a_{" + str(sorted_list[i] + 1) + "}"
        )  # +1 to get the real attribute name
        if (sorted_list[i] + 1) in real_list:
            number_of_correct += 1
    return best_k_features, number_of_correct
