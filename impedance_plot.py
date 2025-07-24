import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy import signal
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help='filename to read data from')
    return parser.parse_args()


def reading(filename):
    df = pd.read_csv(filename, header=5, usecols=[0, 2, 3])
    data = df.iloc[30:-4]  # Input filter
    return data


def dz_dt(df):
    """
    Operations on columns: |Z| = sqrt[ZZ*]
    """
    df['magnitude_sqrt'] = df['BIOZI'] ** 2 + df['BIOZQ'] ** 2
    df['magnitude'] = np.sqrt(df['magnitude_sqrt'])
    # time = registered time - starting time
    # TODO: remove value from first row
    df['time'] = df['timestamp'].astype(float) - 1749475908106
    df['dZ'] = np.ediff1d(df['magnitude'], to_end=0)
    df['dt'] = np.ediff1d(df['time'], to_end=0)
    df['dZ_dt'] = df['dZ'] * (-1) / df['dt']

    return df['dZ_dt']


def plotting(y_column, df):
    """
    Show the plot.
    """
    fig, ax = plt.subplots()
    df['time'] = df['timestamp'].astype(float) - 1749475908106
    ax.plot(df['time'], y_column)
    ax.set(xlabel='time', ylabel='-dZ/dt')
    plt.show()


def denoising(data):
    """
    Denoising of the signal.
    Uses Savitzky-Golay (SG) low-pass digital filter based on "Automatic diagnosis of
    valvular heart diseases by impedance cardiography signal processing".
    :param data: DataFrame
    :return: DataFrame
    Denoised signal.
    """
    # Filter demands no NaNs in the dataframe.
    data = np.nan_to_num(data)
    # Visual comparison of signal denoising on 10 levels of polynomial.
    fig, ax = plt.subplots(10)
    for poly in range(0, 10):
        sig_denoised = signal.savgol_filter(data, 10, poly)
        ax[poly].plot(sig_denoised)
        ax[poly].set_title(poly)
    plt.show()
    # Actual filtering on 2nd degree polynomial.
    chosen_signal = signal.savgol_filter(data, 10, 2)
    return chosen_signal


def squaring(data):
    """
    This step is part of the Pan-Tompkins algorithm.
    https://medium.com/@cosmicwanderer/pan-tompkins-algorithm-for-detecting-qrs-waves-29c5f2927906
    :param data: DataFrame
    :return: DataFrame
    Squared data.
    """
    sq_data = np.power(data, 2)
    fig, ax = plt.subplots(1)
    ax.plot(sq_data)
    plt.title('Squared amplitude')
    plt.show()
    return sq_data


# def moving_average(data, n = 30):
#    window = np.ones((1,n))/n
#    averaging = np. convolve(np.squeeze(data), np.squeeze(window))
#    return averaging


def c_peak_detection(data):
    """
    Detects C-peaks.
    :param data: DataFrame (squared data)
    :return: ndarray
    An array of C-peak times (not values).
    """
    # Detects only C-peaks thanks to signal height threshold.
    peaks, peak_properties = signal.find_peaks(
        data,
        height=np.mean(data) + 10,
        distance=round(1),
    )
    # Visualisation of C-peak locations on squared data.
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.title('C peaks')
    plt.show()
    return np.ndarray.tolist(peaks)


def peak_detection(data, c_peaks_all, denoised):
    """
    Find all peaks on squared data, show plot, for each peak C, peaks B and X are found
    as the second next and previous peak. Visualize peak location on unsquared data.
    :param data: DataFrame
    :param c_peaks: ndarray
    :param denoised: DataFrame
    :return: ndarray
    Arrays with times of peaks.
    """
    peaks, _ = signal.find_peaks(data, height=0, distance=round(1))
    peaks_list = np.ndarray.tolist(peaks)
    # plt.plot(data)
    # plt.plot(peaks, data[peaks], "x")
    # plt.show()
    c_peaks = c_peaks_all[1:-1]
    b_peaks = []
    x_peaks = []
    #iterating through lists and comparing the values
    for c_peak in c_peaks:
         for i, peak in enumerate(peaks_list):
             if c_peak == peak:
                 b_peaks.append(peaks_list[i - 1])
                 x_peaks.append(peaks_list[i + 2])
    b = np.array(b_peaks)
    x = np.array(x_peaks)
    plt.plot(denoised)
    plt.plot(c_peaks, denoised[c_peaks], "x")
    plt.plot(b, denoised[b], "o")
    plt.plot(x, denoised[x], "+")
    plt.title('Peaks')
    plt.show()
    return {
        "c_peaks": c_peaks,
        "b_peaks": b,
        "x_peaks": x
    }


def o_peak_detection(data, denoised, *, c_peaks, b_peaks, x_peaks):
    # TODO: optimize
    peaks, _ = signal.find_peaks(data, height=0, distance=round(1))
    peaks_list = np.ndarray.tolist(peaks)
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.plot(denoised)
    plt.show()
    # print(c_peaks)
    o_peaks = []
    # iterating through lists and comparing the values
    for c_peak in c_peaks:
        for i, peak in enumerate(peaks_list):
            #exception for value 40 (first value on the lists)
            if c_peak == peak and c_peak != 40 and c_peak !=12651:
                o_peaks.append(peaks_list[i+3])
    o = np.array(o_peaks)
    plt.plot(denoised)
    plt.plot(c_peaks, denoised[c_peaks], "x")
    plt.plot(b_peaks, denoised[b_peaks], "o")
    plt.plot(x_peaks, denoised[x_peaks], "+")
    plt.plot(o, denoised[o], "o", color='orange')
    plt.show()
    return {
        "c_peaks": c_peaks,
        "b_peaks": b_peaks,
        "x_peaks": x_peaks,
        "o_peaks": o
    }


def statistical_analysis(denoised, *, c_peaks, b_peaks, x_peaks):
    """
    Mean, standard deviation and variance values table.
    Analized values:
    B-X interval-> systolic ejection time
    B-C interval
    C - systolic peak
    B- beginning of systole
    X- end of systole (dicrotic notch)
    :param denoised: DataFrame (denoised data)
    :param x_peaks: ndarray
    :param b_peaks: ndarray
    :param c_peaks: ndarray
    :return: 0
    Saves file "results.csv".
    """
    #c_peaks = np.delete(c_peaks, 0, 0)
    c_peaks_values = denoised[c_peaks]
    b_peaks_values = denoised[b_peaks]
    x_peaks_values = denoised[x_peaks]
    b_c = np.abs(np.subtract(b_peaks, c_peaks))
    b_x = np.abs(np.subtract(b_peaks, x_peaks))
    # add other statistics and make a table, save to file
    df = pd.DataFrame({
        'Parameter': ['systolic_peak', 'b_peaks', 'dicrotic_notch', 'b_c_interval',
                      'systolic_ejection_time'],
        'Mean': [np.mean(c_peaks_values), np.mean(b_peaks_values),
                 np.mean(x_peaks_values), np.mean(b_c), np.mean(b_x)],
        'Std dev': [np.std(c_peaks_values), np.std(b_peaks_values),
                    np.std(x_peaks_values), np.std(b_c), np.std(b_x)],
        'Variance': [np.var(c_peaks_values), np.var(b_peaks_values),
                     np.var(x_peaks_values), np.var(b_c), np.var(b_x)]})
    df.to_csv('results.csv', index=False)
    return 0


def main():
    args = parse_args()
    data = reading(args.filename)
    signal = dz_dt(data)
    plotting(signal, data)
    denoised = denoising(signal)
    squared = squaring(denoised)
    c_peaks = c_peak_detection(squared)
    peaks_detected = peak_detection(squared, c_peaks, denoised)
    # o_peak_detection(squared, denoised, **peaks_detected)
    statistical_analysis(denoised, **peaks_detected)


if __name__ == '__main__':
    main()
