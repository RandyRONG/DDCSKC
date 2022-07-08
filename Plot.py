
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':
    
    file_name = 'Simulations/COM_2.csv'
    df = pd.read_csv(file_name)
    precisions_cell_RSB = list(df['precisions_cell_RSB'])
    recalls_cell_RSB = list(df['recalls_cell_RSB'])
    
    plt.subplot(1, 2, 1)
    sns.distplot(precisions_cell_RSB, bins=10)
    plt.xlim(0,0.1)
    plt.xlabel('Precison for RSB')
    # ax.set_yticklabels(ax.get_yticks()/len(ARIs_RSB))
    # plt.show()
    
    plt.subplot(1, 2, 2)
    sns.distplot(recalls_cell_RSB, bins=10)
    plt.xlim(0,0.1)
    plt.ylabel('')
    plt.xlabel('Recall for RSB')
    # ax.set_yticklabels(ax.get_yticks()/len(ARIs_RSB))
    plt.show()
    
    
    