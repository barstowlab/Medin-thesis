# functions for helping analyze absorbance data from an AS III Assay

# imports
import numpy as np
import re
from scipy.optimize import curve_fit
from math import nan
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle

# function that reads plate reader absorbance data text file
def read_file(file_name, num_cols, nm_start, nm_end, nm_step):
    all_wells = {}
    for col_reading in range(1,num_cols + 1):
        file = open(file_name)
        nm_match = re.compile(r".*\((\d+)\snm\).*")
        col_match = re.compile(r"(?:\s+\d+){" + str(col_reading - 1) + r"}\s+(\d+)(?:\s+\d+){" + str(num_cols - col_reading) + r"}")
        data_match = re.compile(r"\w(?:\s+\d+[.]\d+){" + str(col_reading - 1) + \
                                r"}\s+(\d+[.]\d+)(?:\s+\d+[.]\d+){" + str(num_cols - col_reading) + r"}")
        nm = 0
        col = 0

        absorbs = np.zeros((8,(nm_end - nm_start)//nm_step + 1))

        cur = 0
        for line in file:
            res = re.match(nm_match, line)
            if res:
                nm = int((int(res.group(1)) - nm_start) / nm_step)
                cur = 0
            res = re.match(col_match, line)
            if res:
                col = int(res.group(1))
            res = re.match(data_match, line)
            if res:
                if col == col_reading:
                    absorbs[cur, nm] = float(res.group(1))
                cur += 1
        
        for r in range(8):
            all_wells[(r+1,col_reading)] = absorbs[r,:]
    return all_wells

# functions for fitting REE content to a single wavelength

# wrapper function for fitting function
def get_absorbance_1fit(ln_conc, *opt):
    return get_absorbance_1(ln_conc, opt)

# fitting function. The parameters opt translates rare earth element concentration 'ln_conc' to absorbance value
def get_absorbance_1(ln_conc, opt):
    A, B, Kd, dye_conc = opt
    b = dye_conc + ln_conc + Kd
    compound_conc = (b - (b*b - 4 * ln_conc * dye_conc)**(0.5))/2
    return A * dye_conc + (B - A) * compound_conc

# calculates REE concentration from absorbance
def get_ree_conc_1(absorbance, opt):
    A, B, Kd, dye = opt
    n1 = (absorbance - A * dye)/(B-A)
    return n1 * (Kd + dye - n1) / (dye - n1)

def get_ree_conc_1fit(ln_conc, *opt):
    return get_ree_conc_1(ln_conc, opt)


# ln_conc should be in ascending order and the first value should be 0
# guesses parameters based on the dataset (see thesis section 2.2.2.2 for thesis guesses
def get_p0_guess_1(ln_conc, abses, dye_conc):
    A0 = abses[0] / dye_conc
    abses = np.array(abses)
    idx = np.argmax(abses[1:] - abses[:len(abses)-1])
    x1 = abses[idx+1] - dye_conc * A0
    x2 = abses[idx] - dye_conc * A0
    a = dye_conc * (ln_conc[idx+1] / x1 - ln_conc[idx] / x2)
    b = -(ln_conc[idx+1] - ln_conc[idx])
    c = x1 - x2
    if b**2 < 4*a*c:
        B1 = A0 + -b / 2 / a
        B2 = A0 + -b / 2 / a
    else:
        B1 = A0 + (-b + np.sqrt(b**2 - 4*a*c)) / 2 / a
        B2 = A0 + (-b - np.sqrt(b**2 - 4*a*c)) / 2 / a
    Kd1 = (dye_conc - x1 / (B1-A0)) * (ln_conc[idx+1] - x1 / (B1-A0)) / (x1 / (B1-A0))
    Kd2 = (dye_conc - x1 / (B2-A0)) * (ln_conc[idx+1] - x1 / (B2-A0)) / (x1 / (B2-A0))
    if Kd2 > 0:
        B0 =B1
        Kd0 = Kd1
    else:
        B0 = B2
        Kd0 = Kd2
    return (min(max(0,A0),1), min(max(0,B0),1), min(1200,max(Kd0, .00001)), dye_conc)

# upper and lower bound for reasonable parameter estimations 
def get_bounds_1(ln_conc, abses, dye_conc):
    A0 = max(.000001,abses[0] / dye_conc)
    A_low = 0.8 * A0
    A_high = 1.2 * A0

    B_low = 0
    B_high = 1
    Kd_low = 0
    Kd_high = 1200
    dye_conc_low = dye_conc * 0.9
    dye_conc_high = dye_conc * 1.1
    lower = (A_low, B_low, Kd_low, dye_conc_low)
    upper = (A_high, B_high, Kd_high, dye_conc_high)
    return (lower, upper)


# -------functions for doing spline fit on multiple wavelengths for single REE-------------#

# does best fit of REE from single wavelength
def get_ree_from_spline(spline, ab, small, big, step):
    poss = np.arange(small, big, step)
    best = small
    best_err = abs(ab -  spline(small))
    for i in range(len(poss)):
        test_val = spline(poss[i])
        if abs(ab - test_val) < best_err:
            best_err = abs(ab - test_val)
            best = poss[i]
    return best

# each entry in map corresponds to ree value in np.arange(small,big,step)
def get_ree_from_spline_fast(spline, ab, small, big, step, abs_map):
    poss = np.arange(small, big, step)
    best = small
    best_err = abs(ab - abs_map[0])
    for i in range(len(poss)):
        test_val = abs_map[i]
        if abs(ab - test_val) < best_err:
            best_err = abs(ab - test_val)
            best = poss[i]
    return best

# returns splines and weights for each wavelength
# abses = [abs_conc1, abs_conc2, ...]
def get_spline_n(ln_conc, abses, nms_idxes, grain=100):
    splines = []
    weights = []
    for idx in nms_idxes:
        ab = []
        for i in range(len(abses)):
            ab.append(abses[i][idx])
        spline = interpolate.UnivariateSpline(ln_conc, ab)
        errs = []
        small = min(ln_conc)
        big = max(ln_conc)
        step = (big - small) / grain
        for i in range(len(ln_conc)):
            errs.append(ln_conc[i] - get_ree_from_spline(spline, ab[i], small, big, step))
        weights.append(1 / np.std(errs))
        splines.append(spline)
    return splines, weights

def get_spline_n_map(splines, small, big, step):
    spline_maps = []
    poss = np.arange(small, big, step)
    for i in range(len(splines)):
        cur_map = np.zeros(len(poss))
        for j in range(len(poss)):
            cur_map[j] = splines[i](poss[j])
        spline_maps.append(cur_map)
    return spline_maps


def get_ree_from_spline_n(splines, weights, abses, small, big, step):
    poss = np.arange(small, big, step)
    best = 0
    best_err = 0
    for i in range(len(splines)):
        best_err += (splines[i](0) - abses[i])**2 * weights[i]**2

    for j in range(len(poss)):
        test_err = 0
        for i in range(len(splines)):
            test_err += (splines[i](poss[j]) - abses[i])**2 * weights[i]**2
        if test_err < best_err:
            best_err = test_err
            best = poss[j]
    return best


def get_ree_from_spline_n_fast(splines, weights, abses, small, big, step, mapping):
    poss = np.arange(small, big, step)
    best = 0
    best_err = 0
    for i in range(len(splines)):
        best_err += (mapping[i][0] - abses[i])**2 * weights[i]**2

    for j in range(len(poss)):
        test_err = 0
        for i in range(len(splines)):
            test_err += (mapping[i][j] - abses[i])**2 * weights[i]**2
        if test_err < best_err:
            best_err = test_err
            best = poss[j]
    return best


#-------functions for doing splines fit on multiple wavelengths for two REEs-------------#

# each abses[i] is an array with the ith wavelength measurements for each REE pair
def get_2ree_spline(ree1s, ree2s, abses, grain=50):
    splines = []
    weights = []
    for i in range(len(abses)):
        splines.append(interpolate.SmoothBivariateSpline(ree1s,ree2s,abses[i]))
        min_ab = min(abses[i])
        max_ab = max(abses[i])
        ab_range = max_ab - min_ab
        errs = []
        for j in range(len(ree1s)):
            ab = abses[i][j]
            ab_guess = splines[-1](ree1s[j], ree2s[j])
            errs.append(((ab - ab_guess) / ab_range)**2)
        weights.append(1 / np.sqrt(np.mean(errs)))
    return splines,weights

def get_2ree_spline_maps(splines, small1, big1, step1, small2, big2, step2):
    maps = []
    for i in range(len(splines)):
        cur_map = {}
        for ree1 in np.arange(small1,big1,step1):
            for ree2 in np.arange(small2,big2,step2):
                cur_map[(ree1,ree2)] = splines[i](ree1,ree2)
        maps.append(cur_map)
    return maps

def get_2ree_from_spline(splines, weights, abses, small1, big1, step1, small2, big2, step2):
    poss1 = np.arange(small1, big1, step1)
    poss2 = np.arange(small2, big2, step2)
    best1 = small1
    best2 = small2
    best_err = 0
    for i in range(len(splines)):
        best_err += abs(abses[i] - splines[i](small1, small2))**2 * weights[i]**2
    for ree1 in poss1:
        for ree2 in poss2:
            err = 0
            for i in range(len(splines)):
                err += abs(abses[i] - splines[i](ree1, ree2))**2 * weights[i]**2
            if err < best_err:
                best_err = err
                best1 = ree1
                best2 = ree2
    return best1, best2

def get_2ree_from_spline_fast(splines, weights, abses, small1, big1, step1, 
    small2, big2, step2, maps):
    poss1 = np.arange(small1, big1, step1)
    poss2 = np.arange(small2, big2, step2)
    best1 = small1
    best2 = small2
    best_err = 0
    for i in range(len(splines)):
        best_err += abs(abses[i] - maps[i][(small1, small2)])**2 * weights[i]**2
    for ree1 in poss1:
        for ree2 in poss2:
            err = 0
            for i in range(len(splines)):
                err += abs(abses[i] - maps[i][(ree1, ree2)])**2 * weights[i]**2
            if err < best_err:
                best_err = err
                best1 = ree1
                best2 = ree2
    return best1, best2


