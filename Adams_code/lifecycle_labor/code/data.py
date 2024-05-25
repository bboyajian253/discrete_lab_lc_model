# lifecycle_labor: data
# import data moments and extract key moments

import numpy as np
import pandas as pd


def import_data(path_data, par):
    """
    imports data moments used in model calibration / validation
    :param path_data: str, path to data files
    :return: mD: dic, container of data moments
    """
    # initialize moment dictionary
    mD = {}

    # import aggregate moments
    df = pd.read_csv(path_data + 'lifetime_stats.csv')

    # store aggregate moments in dictionary
    for v, vv in zip(['ln_ei', 'ln_hAnni', 'ln_lt_hAnni', 'lt_hAnni', 'ln_wi'],
                     ['le', 'lni', 'lnilife', 'nilife', 'leph']
                     ):
        mD[f'mn_{vv}'] = np.array(df[[f'{v}_mean']])[0][0]
        mD[f'sd_{vv}'] = np.array(df[[f'{v}_sd']])[0][0]
    mD['cov_leph_lni'] = (mD['sd_le']**2 - mD['sd_leph']**2 - mD['sd_lni']**2) / 2
    mD['corr_leph_lni'] = mD['cov_leph_lni'] / (mD['sd_leph'] * mD['sd_lni'])

    # import life-cycle moments
    df = pd.read_csv(path_data + 'life_cycle_stats.csv')

    # store life-cycle moments in dictionary
    for v, vv in zip(['ln_ei', 'ln_hAnni', 'ln_wi'], ['le', 'lni', 'leph']):
        temp = df[[f'{v}_mean', f'{v}_sd']]
        temp = np.array(temp)
        mD[f'lc_mn_{vv}'] = temp.T[0]
        mD[f'lc_sd_{vv}'] = temp.T[1]
    mD['lc_cov_le_lni'] = (mD['lc_sd_le'] ** 2 + mD['lc_sd_lni'] ** 2 - mD['lc_sd_leph'] ** 2) / 2
    mD['lc_cov_leph_lni'] = (mD['lc_sd_le'] ** 2 - mD['lc_sd_lni'] ** 2 - mD['lc_sd_leph'] ** 2) / 2
    mD['lc_corr_le_lni'] = mD['lc_cov_le_lni'] / (mD['lc_sd_le'] * mD['lc_sd_lni'])
    mD['lc_corr_leph_lni'] = mD['lc_cov_leph_lni'] / (mD['lc_sd_leph'] * mD['lc_sd_lni'])

    return mD
