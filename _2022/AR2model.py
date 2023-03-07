from random import gauss
import matplotlib.pyplot as plt
from numpy import array


def ar2_model(
        n=10000, initial=[20., 6.],
        a1=.25, a2=.75, const=0, eps=1
) -> list[float]:
    cut = int(n / 10)
    for _ in range(n + cut - len(initial)):
        initial.append(
            round(const
                  + a1 * initial[-1]
                  + a2 * initial[-2]
                  + eps * gauss(mu=0, sigma=1), 13)
        )
    return initial[cut:]


def linear_model(n=10000, k=2., b=0):
    return [k * x + b for x in range(n)]


def generate_model_data_with_plt(
        n=10000, initial=[20., 6.],
        a1=.3, a2=.7, const=0, eps=1
):
    data = ar2_model(n, initial, a1, a2, const, eps)
    # print(data)
    plt.title('1. AR(2)-model')
    plt.plot(data)
    plt.show()
    return array(data)


# generate_model_data_with_plt()
