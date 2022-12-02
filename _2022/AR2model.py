from random import gauss
import matplotlib.pyplot as plt


def ar2_model(
        initial=[20., 6.], a1=0.25, a2=0.75,
        n=10000, const=0, eps=1
) -> list[float]:
    cut = int(n / 10)
    for _ in range(n + cut - len(initial)):
        initial.append(
            const
            + a1 * initial[-1]
            + a2 * initial[-2]
            + eps * gauss(mu=0, sigma=1)
        )
    return initial[cut:]


def linear_model(k=2.0, const=0, n=10000):
    y = [i for i in range(0, n)]
    for i in range(len(y)):
        y[i] = k * y[i] + const
    return y


def generate_model_data_with_plt(initial=[20., 6.], a1=0.3, a2=0.7, n=10000, const=1, eps=1):
    data = ar2_model(initial=initial, a1=a1, a2=a2, n=n, const=const, eps=eps)
    # data = linear_model()
    print(data)
    plt.title('1. AR(2)-model')
    plt.plot(data)
    plt.show()
    return data


# generate_model_data_with_plt()
