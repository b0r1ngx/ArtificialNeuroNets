from matplotlib import pyplot as plt


from _2022.AR2model import generate_model_data_with_plt, ar2_model
from _2022.utils import min_max, normalize_data_between_a_b, denormalize_data, prepare_data


def ar2_model_test():
    n = [20, 50, 100, 1000, 10000]
    datas = []
    for n_i in n:
        data = ar2_model(n=n_i)
        assert n_i == len(data), f'{n_i} != {len(data)}'
        datas.append(data)

    for i in range(1, len(datas)):
        first_el = datas[i - 1][0]
        other_first_el = datas[i][0]
        assert first_el != other_first_el, \
            f'Start is not vary, but must be {first_el} != {other_first_el}'

    plt.plot(data)
    plt.show()
    return datas


def split_data_test(datas):
    values_shrink_at_preparation = 20
    for data in datas:
        assert len(data) >= values_shrink_at_preparation, \
            f'{len(data)} < {values_shrink_at_preparation}'
        prepared = prepare_data(data)
        assert len(data) == len(prepared) + values_shrink_at_preparation, \
            f'{len(data)} != {len(prepared) + values_shrink_at_preparation}'


def normalize_and_denormalize(data=None):
    if data is None:
        data = generate_model_data_with_plt(n=20)

    min, max = min_max(data)
    normalized = normalize_data_between_a_b(data, min, max)
    denormalized = denormalize_data(normalized, min, max)
    for i in range(len(data)):
        init = data[i]
        denorm = denormalized[i]
        assert init == denorm, f'{init} != {denorm}'


def run_all_tests():
    datas = ar2_model_test()
    split_data_test(datas)
    normalize_and_denormalize(datas[-1])


run_all_tests()
