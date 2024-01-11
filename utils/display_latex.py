import matplotlib.pyplot as plt
import numpy as np


def display_score_latex(dimension, row_name, scores_means):
    latex_line = ""
    for i in range(dimension):
        # latex_line+=" & "+str(round(scores_means[i],2))+"-"+str(round(scores_std[i],2))
        latex_line += " & " + str(round(scores_means[i], 3))
    print(row_name + latex_line + " & \\\\")
    return True


def display_feature_latex(dimension, row_name, feature_list, number_of_correct):
    latex_line = ""
    for i in range(10):
        latex_line += " & $" + feature_list[i] + "$"
    print(row_name + latex_line + " & " + str(number_of_correct) + "\\\\")
    return True


# CODE TO PLOT THE DIF(i) SIZE WRT DIMENSION AND SAMPLE SIZE
def display_curve(dif_i_size, sample_size, dimension):
    result = list(np.array(dif_i_size) * (1 / 10))
    print("sample size:", sample_size, "dif_i size", result)
    x = []
    for i in range(dimension):
        x.append(i)
    plt.plot(x, result)
    plt.ylabel("mean Dif(1) size - dim " + str(dimension))
    plt.show()


def prepare_latex(a):
    line = ""
    for i in range(len(a)):
        line = line + " & " + str(a[i])
    return line + "\\"


def show_bar(
    bar1, color1, label1, bar2, color2, label2, x_labels, y_label, legend=True
):
    # width of the bars
    barWidth = 0.3
    x = x_labels
    y_pos1 = np.arange(len(bar1))
    y_pos2 = y_pos1
    # Create bars
    plt.bar(y_pos1, bar1, width=barWidth, color=color1, label=label1)
    plt.bar(y_pos2, bar2, width=barWidth, color=color2, label=label2)
    # Create names on the x-axis
    plt.xticks([r for r in range(len(bar1))], x)
    plt.ylabel(y_label)
    if legend:
        plt.legend()
    # Show graphic
    plt.show()
    plt.savefig("test.png")
    return
