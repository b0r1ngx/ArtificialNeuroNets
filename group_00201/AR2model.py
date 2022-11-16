from random import gauss
import matplotlib.pyplot as plt


def ar2_model(initial=[20., 6.], a1=0.25, a2=0.75, n=12000, c=0, eps=1):
    for _ in range(n - 3):
        initial.append(
            c
            + a1 * initial[-1]
            + a2 * initial[-2]
            + eps * gauss(mu=0, sigma=1)
        )
    return initial[1000:-1000]


def test_model():
    model = ar2_model()
    # print(model)
    plt.plot(model)
    plt.show()


test_model()
