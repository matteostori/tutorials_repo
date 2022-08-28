#main module of the applicaton: it's the entry point to call all the logical blocks of our application
from eda import start_eda
from matplotlib import pyplot as plt

if __name__ == '__main__':
    #1. Exploratory data analysis
    start_eda()

    print('Execution finished')

    # call show here to avoid windows close
    plt.show()