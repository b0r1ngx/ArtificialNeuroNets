import random as r

import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold


# Initial data 1. Noughts and crosses
# Exercise 1,2. Dataset
#
#           ['O', 'O', 'O', 'X',
#           'X', 'X', 'X', 'O',
#           'O', 'X', 'O', 'X',
#           'O', 'X', 'X', 'O']

# My represent of noughts and crosses 'matrix' is:
#       1) each column filled from bottom-left corner to top, and to top-right
#       2) located from [0, 0] to [1, 1] (in coordinate system [x, y])

# Each square of 'matrix':
#       1) starts in the bottom-left corner
#       2) have size a 0.25 measure units
def noughts_n_crosses():
    P = [[0, 0],
         [0, 0.25],
         [0, 0.50],
         [0, 0.75],
         [0.25, 0],
         [0.25, 0.25],
         [0.25, 0.50],
         [0.25, 0.75],
         [0.5, 0],
         [0.5, 0.25],
         [0.5, 0.50],
         [0.5, 0.75],
         [0.75, 0],
         [0.75, 0.25],
         [0.75, 0.50],
         [0.75, 0.75]]

    T = [[0],
         [0],
         [1],
         [0],
         [1],
         [1],
         [1],
         [0],
         [1],
         [0],
         [1],
         [0],
         [0],
         [1],
         [0],
         [1]]

    # Define function. Exercise 2

    # Os - Blues, Xs - Reds
    # Функция покраски тем или иным цветом в зависимости от исходного класса
    # 0-вой класс - нолик
    # 1-вый класс - крестик
    def set_plot_marks(P, T):
        Marks = []
        for i in range(len(P)):
            if T[i][0] == 1:
                Marks.append(Rectangle(P[i], 0.25, 0.25, color="r"))
            elif T[i][0] == 0:
                Marks.append(Rectangle(P[i], 0.25, 0.25, color="b"))
            else:
                print("def set_plot_marks(P, T): Something from dataset (P, T) has not applicable format")
        return Marks

    # Solve. Exercise 2
    # Визуализация примера
    fig, ax = plt.subplots(1)
    plt.title("Xs and Os")
    marks = set_plot_marks(P, T)
    for each in marks:
        ax.add_patch(each)
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_yticks([0.25, 0.5, 0.75])

    # Толстые разделения, для удобности в будущем
    ax.xaxis.grid(True, lw="5")
    ax.yaxis.grid(True, lw="5")

    plt.show()

    return P, T


# uncomment for get all outputs about 1st Task
# noughts_n_crosses()


# Initial data 2. Boolean function
# Exercise 1,2. Dataset
#     f(0, 1, 2, 4, 8, 16, 31) = 1
def boolean_function():
    P = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 1],
         [0, 1, 1, 0, 0],
         [0, 1, 1, 0, 1],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 1, 0],
         [1, 0, 0, 1, 1],
         [1, 0, 1, 0, 0],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 1, 0],
         [1, 0, 1, 1, 1],
         [1, 1, 0, 0, 0],
         [1, 1, 0, 0, 1],
         [1, 1, 0, 1, 0],
         [1, 1, 0, 1, 1],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 1],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]

    T = [[1],
         [1],
         [1],
         [0],
         [1],
         [0],
         [0],
         [0],
         [1],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [1],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [0],
         [1]]

    return P, T


# Определим функцию, инвертирования i-го класса в j-ый
# (работает для использования при работе с <= 10 классами)
def invert_with_k_chance(i, k, number_of_class=2) -> int:
    classes = [i for i in range(number_of_class)]

    if r.random() <= k:
        if i == 0:
            classes.remove(0)
            return r.choice(classes)
        elif i == 1:
            classes.remove(1)
            return r.choice(classes)
        elif i == 2:
            classes.remove(2)
            return r.choice(classes)
        elif i == 3:
            classes.remove(3)
            return r.choice(classes)
        elif i == 4:
            classes.remove(4)
            return r.choice(classes)
        elif i == 5:
            classes.remove(5)
            return r.choice(classes)
        elif i == 6:
            classes.remove(6)
            return r.choice(classes)
        elif i == 7:
            classes.remove(7)
            return r.choice(classes)
        elif i == 8:
            classes.remove(8)
            return r.choice(classes)
        elif i == 9:
            classes.remove(9)
            return r.choice(classes)
    return i


# Определим функцию вычисляющую матрицу неточностей
# Она должна быть следующего вида: к примеру при N = 10 & number_of_classes = 3
# (в нашем случае N больше, различных классов меньше, но суть не меняется)

# if Data:
#   Real: 1 0 2 0 0 1 1 2 2 0
#   Model:1 0 0 1 1 0 2 2 2 2
# then:
#   Confusion Matrix:
#                    0  1  2(model)
#                   ________
#               0  | 1  2  1
#               1  | 1  1  1
#               2  | 1  0  2
#             (real)

# if Data:
#   Real: 0 1 2 3 4 5 0 1 2 3 5 5 2 1 3
#   Model:1 0 2 3 3 4 0 1 2 3 5 2 1 0 5
# then:
#   Confusion Matrix:
#                    0  1  2  3  4  5(model)
#                   _________________
#               0  | 1  1  0  0  0  0
#               1  | 2  1  0  0  0  0
#               2  | 0  1  2  0  0  0
#               3  | 0  0  0  2  0  1
#               4  | 0  0  0  1  0  0
#               5  | 0  0  1  0  1  1
#             (real)
# (функция ниже работает только для работы с 6 классами (но ее не сложно расширить, по схожей логике))
def confusion_matrix(real_data, model_data, number_of_class=2):
    matrix = [[0] * number_of_class for _ in range(number_of_class)]

    for i in range(len(real_data)):
        mData = model_data[i]
        rData = real_data[i][0]

        if type(mData) is np.ndarray:
            mData = mData[0]

        if rData == mData:
            matrix[mData][mData] += 1

        else:
            if mData == 0:
                matrix[rData][0] += 1
            if mData == 1:
                matrix[rData][1] += 1
            elif mData == 2:
                matrix[rData][2] += 1
            elif mData == 3:
                matrix[rData][3] += 1
            elif mData == 4:
                matrix[rData][4] += 1
            elif mData == 5:
                matrix[rData][5] += 1

    if np.sum(matrix) == len(real_data):
        return matrix
    else:
        return "We make bad confusion matrix"


# Создадим функцию выдающую свойства по матрице неточностей,
# чтобы воспользоваться ей как для дву-значащей матрицы класса так и n-классов

# Считать среднюю ошибку распознования будем по формуле (сумма - ср.прав.распозн) / сумма
def find_confusion_matrix_properties(confusion_matrix, number_of_class=6):
    amount_data = np.sum(confusion_matrix)
    avg_correct_recognition = 0

    for i in range(len(confusion_matrix)):
        avg_correct_recognition += confusion_matrix[i][i]

    avg_error_recognition = (amount_data - avg_correct_recognition) / amount_data
    avg_correct_recognition /= amount_data

    if len(confusion_matrix) == 2:
        type_I_error = confusion_matrix[1][0] / amount_data
        type_II_error = confusion_matrix[0][1] / amount_data
        sensitivity = confusion_matrix[1][1] / amount_data
        specificity = confusion_matrix[0][0] / amount_data
        return ["Средняя вероятность ошибки: {0}%".format('%.2f' % (avg_error_recognition * 100)),
                "Средняя вероятность правильного распознования: {0}%".format('%.2f' % (avg_correct_recognition * 100)),
                "Ошибка первого рода: {0}%".format('%.2f' % (type_I_error * 100)),
                "Ошибка второго рода: {0}%".format('%.2f' % (type_II_error * 100)),
                "Чувствительность: {0}%".format('%.2f' % (sensitivity * 100)),
                "Специфичность: {0}%".format('%.2f' % (specificity * 100))]
    else:
        type_I_errors = []
        type_II_errors = []
        for i in range(number_of_class):
            type_I_errors.append((confusion_matrix[0][i] + confusion_matrix[1][i] + confusion_matrix[2][i]
                                  + confusion_matrix[3][i] + confusion_matrix[4][i] + confusion_matrix[5][i]
                                  - confusion_matrix[i][i]) / amount_data)

            type_II_errors.append((confusion_matrix[i][0] + confusion_matrix[i][1] + confusion_matrix[i][2]
                                   + confusion_matrix[i][3] + confusion_matrix[i][4] + confusion_matrix[i][5]
                                   - confusion_matrix[i][i]) / amount_data)

        return ["Средняя вероятность ошибки: {0}%".format('%.2f' % (avg_error_recognition * 100)),
                "Средняя вероятность правильного распознования: {0}%".format('%.2f' % (avg_correct_recognition * 100)),
                "Ошибка первого рода 0-го класса (cyan): {0}%".format('%.2f' % (type_I_errors[0] * 100)),
                "Ошибка второго рода 0-го класса (cyan): {0}%".format('%.2f' % (type_II_errors[0] * 100)),
                "Ошибка первого рода 1-го класса (red): {0}%".format('%.2f' % (type_I_errors[1] * 100)),
                "Ошибка второго рода 1-го класса (red): {0}%".format('%.2f' % (type_II_errors[1] * 100)),
                "Ошибка первого рода 2-го класса (white): {0}%".format('%.2f' % (type_I_errors[2] * 100)),
                "Ошибка второго рода 2-го класса (white): {0}%".format('%.2f' % (type_II_errors[2] * 100)),
                "Ошибка первого рода 3-го класса (blue): {0}%".format('%.2f' % (type_I_errors[3] * 100)),
                "Ошибка второго рода 3-го класса (blue): {0}%".format('%.2f' % (type_II_errors[3] * 100)),
                "Ошибка первого рода 4-го класса (black): {0}%".format('%.2f' % (type_I_errors[4] * 100)),
                "Ошибка второго рода 4-го класса (black): {0}%".format('%.2f' % (type_II_errors[4] * 100)),
                "Ошибка первого рода 5-го класса (yellow): {0}%".format('%.2f' % (type_I_errors[5] * 100)),
                "Ошибка второго рода 5-го класса (yellow): {0}%".format('%.2f' % (type_II_errors[5] * 100))]


# Initial data 3. Dividing the plane into 2 classes
# Exercise 1,2,3,4. Dataset, quality of classification, cross-validation
def plane_2_classes():
    # Exercise 1,2
    # Загружаем данные и получаем цвета в виде [R, G, B, A]
    imginfo = image.imread("../resources/flowers.png")
    fig, ax = plt.subplots()
    ax.imshow(imginfo)
    plt.title("Flowers (input image)")

    # Печатаем наши данные и переходим к следующему заданию
    # Exercise 3. Определение качества классификации
    plt.show()

    # Накопим точки для последуещего отображения классов 0-го и 1-го типов
    def img_data_to_data_sample(task_3_image):
        task_3_x = np.linspace(0, 0.999, 100)
        task_3_P = np.array([(a, b) for a in task_3_x for b in task_3_x])
        task_3_T = np.zeros((len(task_3_P), 3))

        # image size = 450, len(image[each]) = 330
        for it in range(len(task_3_P)):
            y = int(round(task_3_P[it][0] * len(task_3_image))) - 1
            x = int(round(task_3_P[it][1] * len(task_3_image[y])))
            task_3_T[it] = task_3_image[len(task_3_image) - x - 1][y]

        # Black and white color respectively (1 for 100%(255) of rgb)
        task_3_color_a = [1, 1, 1]
        task_3_color_b = [0, 0, 0]

        task_3_T_finalized = np.zeros((len(task_3_T), 1), dtype=int)

        for it in range(len(task_3_T)):
            if np.linalg.norm(task_3_T[it] - task_3_color_a) < 0.15:
                task_3_T_finalized[it] = 0
            elif np.linalg.norm(task_3_T[it] - task_3_color_b) < 0.15:
                task_3_T_finalized[it] = 1
            else:
                # junk
                pass

        def task_3_show():
            a = []
            b = []

            for it in range(len(task_3_T)):
                if task_3_T_finalized[it][0] == 0:
                    a.append(task_3_P[it])
                elif task_3_T_finalized[it][0] == 1:
                    b.append(task_3_P[it])

            a = np.array(a)
            b = np.array(b)

            if len(a) != 0:
                plt.plot(a[:, 0], a[:, 1], 'wo')
            if len(b) != 0:
                plt.plot(b[:, 0], b[:, 1], 'ko')

            plt.title("Input Data Sample")
            plt.show()

        task_3_show()
        # Возвращаем выборку (P, T) размером N (10000), для выполнения последующих заданий
        return task_3_T_finalized

    # Шаг 1
    # Формируем  выборку (P, T) размером N (10000)
    # Для последующих 3-го и 4-го задания данного типа данных
    step_1 = img_data_to_data_sample(imginfo)
    step_1_fixed = img_data_to_data_sample(imginfo).tolist()

    def task_3_show(data, title, cross_validation=False, percent_to_draw_with_other_color=0.0):
        task_3_x = np.linspace(0, 0.999, 100)
        task_3_P = np.array([(a, b) for a in task_3_x for b in task_3_x])

        a = []
        b = []
        c = []
        d = []

        if percent_to_draw_with_other_color == 0.0:
            for it in range(len(data)):
                if data[it] == 0:
                    a.append(task_3_P[it])
                elif data[it] == 1:
                    b.append(task_3_P[it])

            a = np.array(a)
            b = np.array(b)
        else:
            hip = int(len(data) * percent_to_draw_with_other_color)
            for it in range(0, len(data) - hip):
                if data[it] == 0:
                    a.append(task_3_P[it])
                elif data[it] == 1:
                    b.append(task_3_P[it])

            for it in range(len(data) - hip, len(data)):
                if data[it] == 0:
                    c.append(task_3_P[it])
                elif data[it] == 1:
                    d.append(task_3_P[it])

            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            d = np.array(d)

        # wo - white circles, ko - black circles on plot
        if not cross_validation:
            if len(a) != 0:
                plt.plot(a[:, 0], a[:, 1], 'wo')
            if len(b) != 0:
                plt.plot(b[:, 0], b[:, 1], 'ko')
        else:
            if len(a) != 0:
                plt.plot(a[:, 0], a[:, 1], 'wo')
            if len(b) != 0:
                plt.plot(b[:, 0], b[:, 1], 'ko')
            if len(c) != 0:
                plt.plot(c[:, 0], c[:, 1], 'co')
            if len(d) != 0:
                plt.plot(d[:, 0], d[:, 1], 'ro')

        plt.title(title)
        plt.show()

    # Шаг 2
    # Инвентируем метки классов k(5, 10, 20)% для случайно взятых примеров

    # Суть задания и реализации этого в том, что мы как-бы представляем,
    # что это какие-то другие модели (НС) которые распознали или же нет,
    # нашу картинку (данные), полностью и верно, или же с небольшими погрешностями (5, 10, 20)
    # отобразим их, чтобы посмотреть как отработали эти модели.
    step_2_005 = list(map(invert_with_k_chance, step_1, [0.05] * 10000))
    step_2_010 = list(map(invert_with_k_chance, step_1, [0.10] * 10000))
    step_2_020 = list(map(invert_with_k_chance, step_1, [0.20] * 10000))
    task_3_show(step_2_005, "Recognition device k(5)%")
    task_3_show(step_2_010, "Recognition device k(10)%")
    task_3_show(step_2_020, "Recognition device k(20)%")
    # Шаг 3
    # Определяем основные показатели качества распознования:

    # За реальные(желаемые) данные примем данные полученные
    # с помощью нашей интерпритации (на графике Input Data Sample)
    confusion_matrix_for_model_005 = confusion_matrix(step_1_fixed, step_2_005)
    confusion_matrix_for_model_010 = confusion_matrix(step_1_fixed, step_2_010)
    confusion_matrix_for_model_020 = confusion_matrix(step_1_fixed, step_2_020)

    # Определим основные показатели качества распознования Шага 3 (2 класса):
    # Для моделей с 5, 10 и 20-ю вероятностными ошибками в кач-ве распознования данных
    print("Модель с k 5-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m005_properties = find_confusion_matrix_properties(confusion_matrix_for_model_005)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_005:
        print(i)
    for i in m005_properties:
        print(i)
    print()

    print("Модель с k 10-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m010_properties = find_confusion_matrix_properties(confusion_matrix_for_model_010)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_010:
        print(i)
    for i in m010_properties:
        print(i)
    print()

    print("Модель с k 20-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m020_properties = find_confusion_matrix_properties(confusion_matrix_for_model_020)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_020:
        print(i)
    for i in m020_properties:
        print(i)

    # Exercise 4
    # Кросс-валидация
    # Формируем  выборку (P, T) размером N (10000)
    task_4 = step_1

    # Шаги 2, 3:
    # white + black = train set
    # cyan + red = test set
    task_3_show(task_4, "Test and Train Sample", True, 0.50)

    # bad sample of needed K-fold cross-validation
    # task_3_show(task_4, "K-fold cross-validation, k=4", True, 1 / 4)
    # task_3_show(task_4, "K-fold cross-validation, k=8", True, 1 / 8)

    # Пользуемся популярными библиотеками
    kf4 = KFold(n_splits=4, shuffle=True)
    kf8 = KFold(n_splits=8, shuffle=True)

    # Создали функцию, немного другой специфики, специальную для данных,
    # которые приходят из i, j in KFold.split(), в функцию как раз требуется подать train и test данные
    def task_3_show_cv(train, test, title):
        task_3_x = np.linspace(0, 0.999, 100)
        task_3_P = np.array([(a, b) for a in task_3_x for b in task_3_x])

        a = []
        b = []
        c = []
        d = []

        for it in range(len(train)):
            if train[it] == 0:
                a.append(task_3_P[it])
            elif train[it] == 1:
                b.append(task_3_P[it])

        # a little change of how we compose data in arrays c and d here,
        # cos now we have less than Data items (test < Data(train+test)),
        # but we still wanna show test data at the right place
        for it in range(len(test)):
            if test[it] == 0:
                c.append(task_3_P[it + len(train)])
            elif test[it] == 1:
                d.append(task_3_P[it + len(train)])

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        d = np.array(d)

        if len(a) != 0:
            plt.plot(a[:, 0], a[:, 1], 'wo')
        if len(b) != 0:
            plt.plot(b[:, 0], b[:, 1], 'ko')
        if len(c) != 0:
            plt.plot(c[:, 0], c[:, 1], 'co')
        if len(d) != 0:
            plt.plot(d[:, 0], d[:, 1], 'ro')

        plt.title(title)
        plt.show()

    # Здесь, мы можем запечатлить, то, что требовалось по шагу 4, задания 4, указывая выше в kf4, kf8,
    # чтобы данные были перемешены и разделены на k-частей
    for i_train, j_test in kf4.split(task_4):
        task_3_show_cv(task_4[i_train], task_4[j_test], "K-fold cross-validation, k=4")

    for i_train, j_test in kf8.split(task_4):
        task_3_show_cv(task_4[i_train], task_4[j_test], "K-fold cross-validation, k=8")


# uncomment for get all outputs about 3th Task
# plane_2_classes()


# Initial data 4. Dividing the plane into N classes
# Exercise 1,2,3. Dataset, quality of classification
def plane_n_classes():
    # Exercise 1,2
    # Загружаем данные и получаем цвета в виде [R, G, B, A]
    imginfo = image.imread("../resources/owl.png")
    fig, ax = plt.subplots()
    ax.imshow(imginfo)
    plt.title("Owl")

    # Печатаем наши данные и переходим к следующему заданию
    # Exercise 3. Определение качества классификации
    plt.show()

    # Накопим точки для последующего отображения классов n-го типа
    # Для этого, нам нужно понять, со сколькими классами данных мы будем оперировать,
    # взглянем на каринку сосчитаем и определим следующее:
    #       0'class - sky (~blue) (from (150, 200, 255) to (8, 102, 204))
    #       1'class - moon + owl eyes (~yellow/orange) + owl beak (клюв), (from (255, 255, 180) to (255, 237, 0)
    #                                                                                                   + (255, 110, 0))
    #           (sometimes when you look at someone eyes its like you look at moon, like with our owl, so we in)
    #       2'class - tree branch (~brown) (from (113, 75, 61) to (49, 34, 30))
    #       3', 4'classes - owl feathers + a little bit snow on a tree + glare on eyes(~white, dark-blue)
    #       (from (229, 239, 240) to (255), (220) to (125), (170, 190, 200) to (30, 40, 50))
    #       5'class - pupil of owl eye, and all this lines in stained glass (~black) (~10, 30, 30)
    #

    # так же важна, очередность того, как мы будем рисовать каждый цвет
    def img_data_to_data_sample(task_4_image):
        task_4_x = np.linspace(0, 0.999, 450)
        task_4_P = np.array([(a, b) for a in task_4_x for b in task_4_x])
        task_4_T = np.zeros((len(task_4_P), 4))
        junk = []

        for i in range(len(task_4_P)):
            y = int(round(task_4_P[i][0] * len(task_4_image))) - 1
            x = int(round(task_4_P[i][1] * len(task_4_image[y])))
            task_4_T[i] = task_4_image[len(task_4_image) - x - 1][y]

        # create left, mid and right side of each color, for best collect data
        # sky
        task_4_color_a1 = [0.08, 0.39, 0.76, 1.]  # ~ mostly sky dark blue
        task_4_color_a2 = [0.2, 0.67, 0.98, 1.]  # ~ light blue
        task_4_color_a3 = [0.57, 0.79, 0.98, 1.]  # ~ lightly blue

        # moon, eye, beak
        task_4_color_b1 = [0.97, 0.94, 0., 1.]  # ~ yellow
        task_4_color_b2 = [0.98, 0.855, 0.212, 1.]  # ~ yellow
        task_4_color_b3 = [0.995, 0.96, 0.6, 1.]  # ~ light yellow
        task_4_color_b4 = [0.997, 0.93, 0.345, 1.]  # ~ light yellow
        task_4_color_b5 = [0.998, 0.941, 0.404, 1.]  # ~ light yellow
        task_4_color_b6 = [0.98, 0.55, 0.04, 1.]  # ~ orange

        # tree
        task_4_color_c1 = [0.24, 0.13, 0.114, 1.]  # ~ dark brown
        task_4_color_c2 = [0.36, 0.255, 0.22, 1.]  # ~ light brown
        task_4_color_c3 = [0.43, 0.32, 0.275, 1.]  # ~ lightly brown

        # owl feathers
        task_4_color_d1 = [0.5255, 0.5255, 0.5255, 1.]  # ~ gray
        task_4_color_d2 = [0.616, 0.616, 0.616, 1.]  # ~ gray
        task_4_color_d3 = [0.737, 0.737, 0.737, 1.]  # ~ gray
        task_4_color_d4 = [0.81, 0.85, 0.87, 1.]  # ~ dusty white
        task_4_color_d5 = [0.984, 0.98, 0.9725, 1.]  # ~ white

        # owl feathers
        task_4_color_e1 = [0.156, 0.19, 0.22, 1.]  # ~ dark gray-blue
        task_4_color_e2 = [0.208, 0.278, 0.318, 1.]  # ~ dark gray-blue
        task_4_color_e3 = [0.275, 0.36, 0.475, 1.]  # ~ dark gray-blue
        task_4_color_e4 = [0.298, 0.278, 0.22, 1.]  # ~ dark gray-blue
        task_4_color_e5 = [0.302, 0.392, 0.4315, 1.]  # ~ light gray-blue
        task_4_color_e6 = [0.467, 0.557, 0.602, 1.]  # ~ light gray-blue
        task_4_color_e7 = [0.584, 0.65, 0.69, 1.]  # ~ lightly gray-blue

        # frame and lines of stained glass
        task_4_color_f1 = [0.075, 0.075, 0.075, 1.]  # ~ black
        task_4_color_f2 = [0., 0.157, 0.294, 1.]  # ~ blue-black
        task_4_color_f3 = [0.184, 0.153, 0., 1.]  # ~ yellow-black
        task_4_color_f4 = [0.169, 0.149, 0.235, 1.]  # ~ purple-black

        task_4_T_finalized = np.zeros((len(task_4_T), 1), dtype=int)

        for i in range(len(task_4_T)):
            if np.linalg.norm(task_4_T[i] - task_4_color_a1) < 0.02 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_a2) < 0.02 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_a3) < 0.02:
                task_4_T_finalized[i] = 0

            elif np.linalg.norm(task_4_T[i] - task_4_color_b1) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_b2) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_b3) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_b4) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_b5) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_b6) < 0.1:
                task_4_T_finalized[i] = 1

            elif np.linalg.norm(task_4_T[i] - task_4_color_c1) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_c2) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_c3) < 0.1:
                task_4_T_finalized[i] = 2

            elif np.linalg.norm(task_4_T[i] - task_4_color_d1) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_d2) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_d3) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_d4) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_d5) < 0.1:
                task_4_T_finalized[i] = 3

            elif np.linalg.norm(task_4_T[i] - task_4_color_e1) < 0.04 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e2) < 0.04 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e3) < 0.04 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e4) < 0.04 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e5) < 0.08 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e6) < 0.04 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_e7) < 0.04:
                task_4_T_finalized[i] = 4

            elif np.linalg.norm(task_4_T[i] - task_4_color_f1) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_f2) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_f3) < 0.1 \
                    or np.linalg.norm(task_4_T[i] - task_4_color_f4) < 0.1:
                task_4_T_finalized[i] = 5

            else:
                junk.append(task_4_T[i])
                pass

        print('How much dots, goes to junk: ', len(junk))

        def task_4_show(task_4_T, task_4_T_finalized, task_4_P):
            a = []
            b = []
            c = []
            d = []
            e = []
            f = []

            for i in range(len(task_4_T)):
                if task_4_T_finalized[i][0] == 0:
                    a.append(task_4_P[i])

                elif task_4_T_finalized[i][0] == 1:
                    b.append(task_4_P[i])

                elif task_4_T_finalized[i][0] == 2:
                    c.append(task_4_P[i])

                elif task_4_T_finalized[i][0] == 3:
                    d.append(task_4_P[i])

                elif task_4_T_finalized[i][0] == 4:
                    e.append(task_4_P[i])

                elif task_4_T_finalized[i][0] == 5:
                    f.append(task_4_P[i])

            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            d = np.array(d)
            e = np.array(e)
            f = np.array(f)

            if len(a) != 0:
                plt.plot(a[:, 0], a[:, 1], 'c.')  # cyan

            if len(c) != 0:
                plt.plot(c[:, 0], c[:, 1], 'r.')  # red

            if len(d) != 0:
                plt.plot(d[:, 0], d[:, 1], 'w.')  # white

            if len(e) != 0:
                plt.plot(e[:, 0], e[:, 1], 'b.')  # blue

            if len(f) != 0:
                plt.plot(f[:, 0], f[:, 1], 'k.')  # black

            if len(b) != 0:
                plt.plot(b[:, 0], b[:, 1], 'y.')  # yellow

            plt.title("Input Data Sample")
            plt.show()

        task_4_show(task_4_T, task_4_T_finalized, task_4_P)
        # Возвращаем выборку (P, T) размером N (202500), для выполнения последующих заданий
        return task_4_T_finalized

    def task_4_show(data, title):
        task_4_x = np.linspace(0, 0.999, 450)
        task_4_P = np.array([(a, b) for a in task_4_x for b in task_4_x])

        a = []
        b = []
        c = []
        d = []
        e = []
        f = []

        for i in range(len(data)):
            if data[i] == 0:
                a.append(task_4_P[i])

            elif data[i] == 1:
                b.append(task_4_P[i])

            elif data[i] == 2:
                c.append(task_4_P[i])

            elif data[i] == 3:
                d.append(task_4_P[i])

            elif data[i] == 4:
                e.append(task_4_P[i])

            elif data[i] == 5:
                f.append(task_4_P[i])

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        d = np.array(d)
        e = np.array(e)
        f = np.array(f)

        if len(a) != 0:
            plt.plot(a[:, 0], a[:, 1], 'c.')  # cyan

        if len(c) != 0:
            plt.plot(c[:, 0], c[:, 1], 'r.')  # red

        if len(d) != 0:
            plt.plot(d[:, 0], d[:, 1], 'w.')  # white

        if len(e) != 0:
            plt.plot(e[:, 0], e[:, 1], 'b.')  # blue

        if len(f) != 0:
            plt.plot(f[:, 0], f[:, 1], 'k.')  # black

        if len(b) != 0:
            plt.plot(b[:, 0], b[:, 1], 'y.')  # yellow

        plt.title(title)
        plt.show()

    # Шаг 1
    # Формируем  выборку (P, T) размером N (202500)
    # Для последуюшего 3-го задания данного типа данных
    step_1 = img_data_to_data_sample(imginfo)

    # Шаг 2
    # Инвертируем метки классов k(5, 10, 20)% для случайно взятых примеров
    step_2_005 = list(map(invert_with_k_chance, step_1, [0.05] * 202500, [6] * 202500))
    step_2_010 = list(map(invert_with_k_chance, step_1, [0.10] * 202500, [6] * 202500))
    step_2_020 = list(map(invert_with_k_chance, step_1, [0.20] * 202500, [6] * 202500))

    # смотря, в первый раз на изображения, которые мы получем здесь, можно подумать, что в наших данных ошибка,
    # покрайней мере я смутился от этого, но потом вновь вспомнил, что очень важно какой именно цвет, в конце мы рисуем
    # тот и будет в большинстве, потому что просто начинает перекрывать, остальные точки, которые лежат ниже, так что,
    # в эти функции можно добавить элемент, который будет рисовать последним цветом наугад.

    # видим последний цвет не изменненным, так как его больше всего
    task_4_show(step_2_005, "Recognition device k(5)%")
    task_4_show(step_2_010, "Recognition device k(10)%")
    task_4_show(step_2_020, "Recognition device k(20)%")

    # Шаг 3
    # Определяем основные показатели качества распознования:

    # За реальные(желаемые) данные примем данные полученные
    # с помощью нашей интерпритации (на графике Input Data Sample)
    confusion_matrix_for_model_005 = confusion_matrix(step_1, step_2_005, 6)
    confusion_matrix_for_model_010 = confusion_matrix(step_1, step_2_010, 6)
    confusion_matrix_for_model_020 = confusion_matrix(step_1, step_2_020, 6)

    # Определим основные показатели качества распознования Шага 3 (2 класса):
    # Для моделей с 5, 10 и 20-ю вероятностными ошибками в кач-ве распознования данных
    print("Модель с k 5-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m005_properties = find_confusion_matrix_properties(confusion_matrix_for_model_005)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_005:
        print(i)
    for i in m005_properties:
        print(i)
    print()

    print("Модель с k 10-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m010_properties = find_confusion_matrix_properties(confusion_matrix_for_model_010)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_010:
        print(i)
    for i in m010_properties:
        print(i)
    print()

    print("Модель с k 20-ью% ошибками")
    print("Убедимся в том, что средняя вер-ть ошибки примерно совпадает с заданным k%")
    m020_properties = find_confusion_matrix_properties(confusion_matrix_for_model_020)
    print("Матрица неточностей")
    for i in confusion_matrix_for_model_020:
        print(i)
    for i in m020_properties:
        print(i)


# uncomment for get all outputs about 4th Task
# plane_n_classes()


# Initial data 5. Continuous function of one variable
# Exercise 1,2,3. Define function (dataset), quality of approximation
def continuous_function():
    # Exercise 2
    # Step 2. Функция принимающая и выдающая значения
    def f(P):
        return np.sin(1.3 * P) + np.cos(37 * P)

    # Step 3. Узлы для построения (Мн-во  входных значений P, в диапазоне [-3, 3])
    N = 3000
    P5 = np.linspace(-3, 3, N)

    # Некоторые настройки для отображения графика
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')

    # Step 4, 5. Определение значений T и визуализация выборки
    plt.title("y = sin(1.3x) + cos(37x)")
    plt.plot(P5, f(P5), 'r')
    plt.show()

    # Exercise 3. Определение качества аппроксимации
    # Step 1. Формирование выборты (P, T), объемом N = 3000
    x = P5
    y = f(P5)

    y_max = max(y)  # ~ 2
    y_min = min(y)
    size = N  # 3000

    noise_005 = .05 * y_max
    noise_010 = .1 * y_max
    noise_020 = .2 * y_max

    def noise_with_amplitude(y, noise_level, N=3000):
        noise = np.random.uniform(-noise_level, noise_level, N)
        noised = []
        for i in range(N):
            noised.append(y[i] + noise[i])
        return noised, sum(noise) / N, max(noise)

    noised_y_005, rel_005, mme_005 = noise_with_amplitude(y, noise_005)
    noised_y_010, rel_010, mme_010 = noise_with_amplitude(y, noise_010)
    noised_y_020, rel_020, mme_020 = noise_with_amplitude(y, noise_020)
    print('Средняя абсолютная ошибка')
    mae_005 = sum(abs(y - noised_y_005)) / N
    mae_010 = sum(abs(y - noised_y_010)) / N
    mae_020 = sum(abs(y - noised_y_020)) / N
    print('шум 5% - ', mae_005)
    print('шум 10% - ', mae_010)
    print('шум 20% - ', mae_020)
    print('Средняя относительная ошибка')
    # тут какая-то ерунда получается, видимо я ерунду какую-то и считаю, MRE = MAPE?
    mre_005 = sum(abs(y - rel_005)) / N
    mre_010 = sum(abs(y - rel_010)) / N
    mre_020 = sum(abs(y - rel_020)) / N
    mre_005 = sum(abs((y - noised_y_005) / y)) / N
    mre_010 = sum(abs((y - noised_y_010) / y)) / N
    mre_020 = sum(abs((y - noised_y_020) / y)) / N
    print('шум 5% - ', mre_005)
    print('шум 10% - ', mre_010)
    print('шум 20% - ', mre_020)
    print('Максимальная по модулю ошибка')
    # не совсем понял, что это должно быть:
    # максимальная ошибка (между максимальным значением в функции и максимальным шумовым значением)
    # или максимальной ошибкой по модулю, которую я по своему мнению и вывел

    # mme_005 = abs(max(y) - max(noised_y_005))
    # mme_010 = abs(max(y) - max(noised_y_010))
    # mme_020 = abs(max(y) - max(noised_y_020))

    # так как значение амлитудное, здесь видим двойное значени, 10, 20 и 40 соответственно
    print('шум 5% - ', mme_005)
    print('шум 10% - ', mme_010)
    print('шум 20% - ', mme_020)

    plt.title("noised 5%")
    plt.plot(x, noised_y_005, 'b')
    plt.show()
    plt.title("noised 10%")
    plt.plot(x, noised_y_010, 'y')
    plt.show()
    plt.title("noised 20%")
    plt.plot(x, noised_y_020, '-k')
    plt.show()
    print('По выведенным данным выше, убеждаемся, что значения ошибок, соответствуют заданным шумовым значениям')

# uncomment for get all outputs about 5th Task
# continuous_function()
