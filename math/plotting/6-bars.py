#!/usr/bin/env python3
'''
Module to plot a stacked bar graph of fruit distribution
'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''
    Plots a stacked bar graph with specific colors, widths, and ticks
    '''

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    persons = ['Farrah', 'Fred', 'Felicia']
    width = 0.5

    plt.bar(persons, apples, width=width, color='red', label='apples')

    plt.bar(persons, bananas, width=width, bottom=apples,
            color='yellow', label='bananas')

    plt.bar(persons, oranges, width=width, bottom=apples + bananas,
            color='#ff8000', label = 'oranges')

    plt.bar(persons, peaches, width=width, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')

    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    plt.legend()

    plt.show()
