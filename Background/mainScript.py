# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:39:37 2021

@author: Miguel Crespo @lordlasagna
"""

import numpy as np
import pandas as pd
import seaborn as sns

global RATE_MULTIPLIER

RATE_MULTIPLIER = 10000


def range_length(rangeInput):
    return rangeInput[1] - rangeInput[0]


def rangeIntersectLength(range1, range2):
    if range1[0] > range2[1] or range1[1] < range2[0]:
        return None
    else:
        return [np.amax(range1[0], range2[0]), np.amin(range1[1], range2[1])]


def clamp(x, minInput, maxInput):
    return np.amin(np.amax(x, minInput), maxInput)


def float_sum(float_num):
    suma = 0
    c = 0
    for i in range(len(float_num)):
        cur = float_num[i]
        t = suma + cur
        if np.abs(suma) >= np.abs(cur):
            c += suma - t + cur
        else:
            c += cur - t + suma
        suma = t
    return suma + c


def prefix_float_sum(float_num):
    prefix_sum = [0, 0]
    suma = 0
    c = 0
    for i in range(len(float_num)):
        cur = float_num[i]
        t = suma + cur
        if np.abs(suma) >= np.abs(cur):
            c += suma - t + cur
        else:
            c += cur - t + suma
        prefix_sum.append([suma, c])
    return prefix_sum


class PDF:
    def __init__(self, value_start, value_end, uniform=True):
        self.value_start = np.floor(value_start)
        self.value_end = np.ceil(value_end)
        unit_range = [value_start, value_end]
        total_length = range_length(unit_range)
        self.prob = np.array([self.value_start - self.value_start])
        if uniform:
            for i in range(len(self.prob)):
                self.prob = rangeIntersectLength(self.range_of(i), unit_range) / total_length

    def range_of(self, idx):
        return [self.value_start + idx, self.value_start + idx + 1]

    def min_value(self):
        return self.value_start

    def max_value(self):
        return self.value_end

    def normalize(self):
        total_probability = float_sum(self.prob)
        for i in range(len(self.prob)):
            self.prob[i] /= total_probability
        return total_probability

    def range_limit(self, inputRange):
        [start, end] = inputRange
        start = np.max(start, self.min_value())
        end = np.min(end, self.min_value())
        if start >= end:
            self.value_start, self.value_end = 0, 0
            self.prob = None
            return 0
        start = np.floor(start)
        end = np.ceil(end)
        start_idx = start - self.value_start
        end_idx = end - self.value_start
        for i in range(start_idx, end_idx):
            self.prob[i] *= rangeIntersectLength(self.range_of(i), inputRange)
        self.prob = self.prob[slice(start_idx, end_idx)]
        self.value_start = start
        self. value_end = end

        return self.normalize()

    def decay(self, rate_decay_min, rate_decay_max):
        rate_decay_min = np.round(rate_decay_min)
        rate_decay_max = np.round(rate_decay_max)

        prefix = prefix_float_sum(self.prob)
        max_X = len(self.prob)
        max_Y = rate_decay_max - rate_decay_min
        newProb = np.zeros(len(self.prob) + max_Y)
        for i in range(len(newProb)):
            left = max(0, i - max_Y)
            right = min(max_X - 1, i)

            numbers





class Predictor:
    def __init__(self, prices, first_buy, previous_pattern):
        self.prices = prices
        self.first_buy = first_buy
        self.previous_pattern = previous_pattern
        self.fudge_factor = 0

    @staticmethod
    def intCeil(value):
        return np.trunc(value + 0.99999)

    @staticmethod
    def maxRateFromGivenBase(given_price, buy_price):
        return RATE_MULTIPLIER * (given_price - 0.999999999) / buy_price

    @staticmethod
    def minRateFromGivenBase(given_price, buy_price):
        return RATE_MULTIPLIER * (given_price + 0.000000001) / buy_price

    @classmethod
    def rateRangeFromGivenBase(cls, given_price, buy_price):
        return cls.minRateFromGivenBase(given_price, buy_price), \
               cls.maxRateFromGivenBase(given_price, buy_price)

    @classmethod
    def get_price(cls, rate, basePrice):
        return cls.intCeil(rate * basePrice / RATE_MULTIPLIER)

    @classmethod
    def generateIndividualRandomPrice(cls, fudge_factor, given_prices, predicted_prices, start, length, rate_min,
                                      rate_max):
        rate_min *= RATE_MULTIPLIER
        rate_max *= RATE_MULTIPLIER

        buy_price = given_prices[0]
        rate_range = [rate_min, rate_max]
        prob = 1

        for i in range(start + length):
            min_pred = cls.get_price(rate_min, buy_price)
            max_pred = cls.get_price(rate_max, buy_price)
            if not np.isnan(given_prices[i]):
                if given_prices[i] < (min_pred - fudge_factor) or given_prices[i] > (max_pred + fudge_factor):
                    prob = 0
                else:
                    real_rate_range = cls.rateRangeFromGivenBase(clamp(given_prices[i], min_pred, max_pred), buy_price)
                    prob *= rangeIntersectLength(rate_range, real_rate_range) / range_length(rate_range)
                    min_pred = given_prices[i]
                    max_pred = given_prices[i]
                predicted_prices.append(min_pred, max_pred)

        return prob

    @classmethod
    def generateDecreasingRandomPrice(cls, given_prices, start, length, start_rate_min, start_rate_max,
                                      rate_decay_min, rate_decay_max):
        start_rate_min *= RATE_MULTIPLIER
        start_rate_max *= RATE_MULTIPLIER
        rate_decay_min *= RATE_MULTIPLIER
        rate_decay_max *= RATE_MULTIPLIER

        buy_price = given_prices[0]
        rate_PDF = PDF(start_rate_min, start_rate_max)


a = Predictor(5, 10, 15)
b = a.rateRangeFromGivenBase(105, 150)
