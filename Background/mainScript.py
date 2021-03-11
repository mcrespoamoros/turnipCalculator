# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:39:37 2021

@author: Miguel Crespo @lordlasagna
"""

import numpy as np
import pandas as pd
import seaborn as sns
from math import trunc

global RATE_MULTIPLIER

RATE_MULTIPLIER = 10000


class Predictor:
    def __init__(self, prices, first_buy, previous_pattern):
        self.prices = prices
        self.first_buy = first_buy
        self.previous_pattern = previous_pattern

    @classmethod
    def intCeil(cls, value):
        return trunc(value + 0.99999)

    @classmethod
    def maxRateFromGivenBase(cls, given_price, buy_price):
        return RATE_MULTIPLIER * (given_price - 0.999999999) / buy_price

    @classmethod
    def minRateFromGivenBase(cls, given_price, buy_price):
        return RATE_MULTIPLIER * (given_price + 0.000000001) / buy_price

    @classmethod
    def rateRangeFromGivenBase(cls, given_price, buy_price):
        return cls.minRateFromGivenBase(given_price, buy_price), \
               cls.maxRateFromGivenBase(given_price, buy_price)

    @classmethod
    def get_price(cls, rate, basePrice):
        return cls.intCeil(rate * basePrice / RATE_MULTIPLIER)

    @classmethod
    def generateIndividualRandomPrice(cls, given_prices, predicted_prices, start, length, rate_min, rate_max):
        rate_min *= RATE_MULTIPLIER
        rate_max *= RATE_MULTIPLIER
        



a = Predictor(5, 10, 15)
b = a.rateRangeFromGivenBase(105, 150)
