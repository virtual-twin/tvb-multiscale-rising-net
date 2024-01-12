# -*- coding: utf-8 -*-

def multiply(x, y):
    output = print(x*y)
    return output


def condition(x, th=100.0):
    if x <= th:
        output = 1
    else:
        output = 0
    print(output)
    return output


if __name__ == '__main__':
    import sys

    if sys.argv[1] == "multiply":
        multiply(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "condition":
        x = float(sys.argv[2])
        if len(sys.argv) > 3:
            condition(x, float(sys.argv[3]))
        else:
            condition(x)
