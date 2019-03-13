import matplotlib.pyplot as plt
import h5py
import os

def parse_evaluations(base_dir, setups=[], iteration=3*10**5, run=0):
    setup_strings = []

    for s in setups:
        s_str = str(s)
        if len(s_str) == 1:
            s_str = "0" + s_str

        setup_strings.append(s_str)
            

    max_evals = [os.path.join(base_dir, "setup" + s + "/{}/run_{}/evaluation_maxima_lsds_validation_B+.h5".format(iteration, run)) for s in setup_strings]

    evaluation = {}
    for f_eval, s in zip(max_evals, setup_strings):
        if os.path.isfile(f_eval):
            with h5py.File(f_eval, "r") as f:
                fp = f["fp"].value
                fn = f["fn"].value
                tp = f["tp"].value

                evaluation[s] = [tp, fp, fn]

    return evaluation


def plot_f1(evaluation):

    plt.figure()
    for setup, score in evaluation.items():
        precision = float(score[0])/(score[0] + score[1])
        recall = float(score[0])/(score[0] + score[2])
        plt.scatter(precision, recall)
        plt.text(precision, recall, setup)
        plt.xlabel("Precision")
        plt.ylabel("Recall")

    plt.show()

if __name__ == "__main__":
    evaluation = parse_evaluations("/groups/funke/home/ecksteinn/Projects/microtubules/micronet/micronet/cremi/03_predict",
                                  [1])

    plot_f1(evaluation)






