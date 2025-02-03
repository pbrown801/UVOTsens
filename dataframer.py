"""
This module contains the `process_photometry_data` function which processes photometry data files
and returns a DataFrame containing measurements and computed aperture corrections.
"""

import os
import numpy as np
import pandas as pd

#================================#
#   Helper functions (Private)   #
#================================#

def __initialize_arrays(shape, count):
    return [np.full(shape, np.nan, dtype='float') for _ in range(count)]


def __ap_cor(cr5, cr3):
    return cr5 / cr3 if cr3 != 0 else np.nan


def __ap_cor_err(cr5, cr3, cr5e, cr3e):
    return np.sqrt((cr5e/cr5)**2 + (cr3e/cr3)**2) * __ap_cor(cr5, cr3)

#========================#
#    Public Functions    #
#========================#

def filter_entries(df, filters=None, fields=None, comps=None, epochs=None, entries=None, comparison='>', threshold=1.4, print_summary=False):
    """
    Filter entries in the DataFrame based on filter, field, comp, and epoch values.
    Threshold is used to filter the entries.
    """

    if not entries:
        return df

    if filters:
        df = df[df['filter'].isin(filters)]
    if fields:
        df = df[df['field'].isin(fields)]
    if comps:
        df = df[df['comp'].isin(comps)]
    if epochs:
        df = df[df['epoch'].isin(epochs)]
    if threshold:
        for entry in entries:
            if comparison == '>':
                df = df[df[entry] > threshold]
            elif comparison == '<':
                df = df[df[entry] < threshold]
            elif comparison == '==':
                df = df[df[entry] == threshold]
            elif comparison == '!=':
                df = df[df[entry] != threshold]
            elif comparison == '>=':
                df = df[df[entry] >= threshold]
            elif comparison == '<=':
                df = df[df[entry] <= threshold]
            else:
                raise ValueError(f"Invalid comparison operator: {comparison}")

    if print_summary:
        print(f"Number of entries: {len(df)}")
        print(f"Number of filters: {len(df['filter'].unique())}")
        print(f"Number of fields: {len(df['field'].unique())}")
        print(f"Number of comps: {len(df['comp'].unique())}")
        print(f"Number of epochs: {len(df['epoch'].unique())}")

    return df

def find_high_apcor_entries(df, threshold=1.4):
    """
    Find entries in the DataFrame with aperture correction errors above a threshold.
    """
    return df[(df['apcore'] > threshold) | (df['apcore_centroid'] > threshold)]

def process_photometry_data(filters, fields, comps, main_file_pattern, centroid_file_pattern, mostepochs=300):
    """
    Process photometry data files and return a DataFrame containing
    measurements and computed aperture corrections.
    """
    mostcomps = len(comps)
    mostfilters = len(filters)
    mostfields = len(fields)

    # --- Initialize arrays with NaN.
    shape1 = (mostfilters, mostfields, mostcomps, mostepochs)
    shape2 = (mostfilters, mostfields, mostepochs)
    shape3 = (mostfilters, mostfields, mostcomps)
    year, exposure, telapse, bkg, cr3, cr3e, cr5, cr5e = __initialize_arrays(shape1, 8)
    cr3_centroid, cr3e_centroid, cr5_centroid, cr5e_centroid = __initialize_arrays(shape1, 4)
    apcor_centroid, apcore_centroid, apcor, apcore = __initialize_arrays(shape1, 4)
    frameapcor, frameapcorscat = __initialize_arrays(shape2, 2)
    compapcor, compapcorscat = __initialize_arrays(shape3, 2)

    # --- Loop through all files to fill the arrays.
    for i, filt in enumerate(filters):
        for j, field in enumerate(fields):
            for c, comp in enumerate(comps):
                file = main_file_pattern.format(field=field, filter=filt, comp=comp)
                filecentroid = centroid_file_pattern.format(field=field, filter=filt, comp=comp)

                # Process the main file using `with` to ensure resources are released.
                # Additionally, check the column size to avoid accessing invalid memory.
                if os.path.exists(file):
                    with open(file, 'r', encoding='utf-8') as fhandle:
                        for e, line in enumerate(fhandle):
                            columns = line.split()
                            if len(columns) < 9:
                                continue
                            year[i, j, c, e]     = float(columns[1])
                            exposure[i, j, c, e] = float(columns[2])
                            telapse[i, j, c, e]  = float(columns[3])
                            cr3[i, j, c, e]      = float(columns[4])
                            cr3e[i, j, c, e]     = float(columns[5])
                            bkg[i, j, c, e]      = float(columns[6])
                            cr5[i, j, c, e]      = float(columns[7])
                            cr5e[i, j, c, e]     = float(columns[8])

                    # --- Data cleaning
                    # Note: In the original code, the `reject_outliers` function is used,
                    # but the values were not used afterwards. I've commented it out.
                    if np.nanmedian(cr5[i, j, c, :]) > 150:
                        cr5[i, j, c, :] = np.nan
                    cr5[cr3 < 0] = np.nan
                    # cleaneddata = reject_outliers(cr5[i, j, c, :], m=2)

                # Process the centroid file (if it exists) to fill the centroid arrays.
                if os.path.exists(filecentroid):
                    with open(filecentroid, 'r', encoding='utf-8') as fhandle:
                        for e, line in enumerate(fhandle):
                            columns = line.split()
                            if len(columns) < 9:
                                continue
                            cr3_centroid[i, j, c, e]  = float(columns[4])
                            cr3e_centroid[i, j, c, e] = float(columns[5])
                            cr5_centroid[i, j, c, e]  = float(columns[7])
                            cr5e_centroid[i, j, c, e] = float(columns[8])
                            # Compute the centroid aperture correction using the values from the main file.
                            if cr3[i, j, c, e] != 0:
                                apcor_centroid[i, j, c, e] = __ap_cor(cr5_centroid[i, j, c, e], cr3_centroid[i, j, c, e])
                            else:
                                apcor_centroid[i, j, c, e] = np.nan
                            # Propagate the error
                            if cr5[i, j, c, e] != 0 and cr3[i, j, c, e] != 0:
                                apcore_centroid[i, j, c, e] = __ap_cor_err(cr5_centroid[i, j, c, e], cr3_centroid[i, j, c, e],
                                                                            cr5e_centroid[i, j, c, e], cr3e_centroid[i, j, c, e])
                            else:
                                apcore_centroid[i, j, c, e] = np.nan

                    # Compute per-star aperture correction (mean and scatter) over epochs.
                    ratio = np.divide(cr5[i, j, c, :], cr3[i, j, c, :], out=np.full_like(cr5[i, j, c, :], np.nan), where=cr3[i, j, c, :]!=0)
                    compapcor[i, j, c]    = np.nanmean(ratio)
                    compapcorscat[i, j, c] = np.nanstd(ratio)

            # S/N (signal-to-noise) cuts.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_cr3 = np.divide(cr3, cr3e, out=np.full_like(cr3, np.nan), where=cr3e!=0)
                ratio_cr5 = np.divide(cr5, cr5e, out=np.full_like(cr5, np.nan), where=cr5e!=0)
            cr3[ratio_cr3 < 5] = np.nan
            cr5[ratio_cr5 < 5] = np.nan

            # Compute frame-level aperture corrections (mean and scatter) over comps for each epoch.
            for e in range(0, mostepochs - 2):
                ratios = np.divide(cr5[i, j, :, e], cr3[i, j, :, e], out=np.full((mostcomps,), np.nan), where=cr3[i, j, :, e]!=0)
                frameapcor[i, j, e]    = np.nanmean(ratios)
                frameapcorscat[i, j, e] = np.nanstd(ratios)

    with np.errstate(divide='ignore', invalid='ignore'):
        apcor = np.divide(cr5, cr3, out=np.full_like(cr5, np.nan), where=cr3!=0)
    apcore = np.sqrt((np.divide(cr5e, cr5, out=np.full_like(cr5, np.nan), where=cr5!=0))**2 +
                     (np.divide(cr3e, cr3, out=np.full_like(cr3, np.nan), where=cr3!=0))**2) * apcor

    # S/N cuts and negative cr3 values for the aperture corrections.
    for arr in [apcor, apcore, apcor_centroid, apcore_centroid]:
        arr[ratio_cr3 < 5] = np.nan
        arr[cr3 < 0] = np.nan
    # High error cut.
    apcor[apcore > 0.2] = np.nan
    apcore[apcore > 0.2] = np.nan
    apcor_centroid[apcore > 0.2] = np.nan
    apcore_centroid[apcore > 0.2] = np.nan

    # --- Create the DataFrame.
    num_filters, num_fields, num_comps, num_epochs = year.shape

    # Create a MultiIndex for the flattened arrays.
    index = pd.MultiIndex.from_product(
        [range(num_filters), range(num_fields), range(num_comps), range(num_epochs)],
        names=['filter_idx', 'field_idx', 'comp_idx', 'epoch']
    )

    # Flatten the arrays so they can be used in a DataFrame.
    df = pd.DataFrame({
        'year':             year.flatten(),
        'exposure':         exposure.flatten(),
        'telapse':          telapse.flatten(),
        'bkg':              bkg.flatten(),
        'cr3':              cr3.flatten(),
        'cr3e':             cr3e.flatten(),
        'cr5':              cr5.flatten(),
        'cr5e':             cr5e.flatten(),
        'apcor':            apcor.flatten(),
        'apcore':           apcore.flatten(),
        'cr3_centroid':     cr3_centroid.flatten(),
        'cr3e_centroid':    cr3e_centroid.flatten(),
        'cr5_centroid':     cr5_centroid.flatten(),
        'cr5e_centroid':    cr5e_centroid.flatten(),
        'apcor_centroid':   apcor_centroid.flatten(),
        'apcore_centroid':  apcore_centroid.flatten()
    }, index=index).reset_index()

    # Map numeric indices to actual filter, field, and comp names.
    # I.e., replace the numeric index columns with the actual names.
    # Otherwise, it'd show up as 0, 1, 2, etc. instead of 'B', 'V', 'R', etc.
    df['filter'] = df['filter_idx'].map(lambda i: filters[i])
    df['field']  = df['field_idx'].map(lambda i: fields[i])
    df['comp']   = df['comp_idx'].map(lambda i: comps[i])

    # Drop the numeric index columns.
    # We don't need them anymore since we have the actual names.
    df.drop(['filter_idx', 'field_idx', 'comp_idx'], axis=1, inplace=True)

    # Remove rows where data were never filled (e.g., year remains NaN).
    # The reason for this is that due to creating the arrays with a fixed size,
    # there can be rows where the data was never filled (for some reason or another).
    # So, remove those rows to avoid having NaN rows in the final DataFrame.
    df = df.dropna(subset=['year'])

    return df
