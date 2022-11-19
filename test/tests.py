from matplotlib import pyplot as plt

from group_00201.AR2model import ar2_model
from group_00201.utils import split_data


def ar2_model_test():
    n = [20, 50, 100, 1000, 10000]
    datas = []
    for n_i in n:
        data = ar2_model(n=n_i)
        assert n_i == len(data)
        datas.append(data)

    # check that start is vary
    for i in range(1, len(datas)):
        assert datas[i-1][0] != datas[i][0]

    plt.plot(data)
    plt.show()
    return datas


def split_data_test(datas):
    values_shrink_at_split_data = 20
    for data in datas:
        assert len(data) >= values_shrink_at_split_data
        split = split_data(data)
        assert len(data) == len(split) + values_shrink_at_split_data


def run_all_tests():
    datas = ar2_model_test()
    split_data_test(datas)


run_all_tests()
