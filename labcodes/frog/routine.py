"""Script provides functions dealing with routine experiment datas."""


import numpy as np
from labcodes import misc, fitter, models


def prep_data_one(logf, atten=0):
    """Normalize S21 data for models.ResonatorModel_inverse."""
    df = logf.df.copy()
    angle = misc.remove_e_delay(df['s21_rad'].values, df['freq_GHz'].values)
    df['s21'] = 10 ** (df['s21_dB'] / 20) * np.exp(1j*angle)
    df['s21'] = models.ResonatorModel_inverse.normalize(df['s21'])
    df['1_s21'] = 1/df['s21']
    df['id'] = int(logf.name[:5])
    # df['bw_Hz'] = float(logf.config['Parameter 2']['data'][6:-7])
    df['bw_kHz'] = float(logf.config['Parameter 2']['data'][6:-8])
    df['power_dBm'] = float(logf.config['Parameter 3']['data'][6:-8]) - atten
    df['frr_GHz'] = df['freq_GHz'].mean()
    return df

def fit_resonator(logf, atten=0, **kwargs):
    df = prep_data_one(logf, atten)
    cfit = fitter.CurveFit(
        xdata=df['freq_GHz'].values,
        ydata=df['1_s21'].values,
        model=models.ResonatorModel_inverse(),
        hold=True,
    )
    cfit.fit(**kwargs)
    ax = cfit.plot_complex(plot_init=False, fit_report=True)
    return cfit, ax
    
def fit_t1(logf, ax, xy_text, unit='\\mu s', t2e=False):
    cfit = fitter.CurveFit(
        xdata=logf.df['delay_us'].values,
        ydata=logf.df['s1_prob'].values,
        model=models.ExponentialModel(),
    )
    ax.plot(cfit.xdata, cfit.fdata(), 'r-', lw=1)
    tau = cfit.result.params["tau"].value
    tau_err = cfit.result.params["tau"].stderr
    if t2e is False:
        ax.text(xy_text[0], xy_text[1], 
            f'$T_1\\approx {tau:.2f}\\pm{tau_err:.4f} {unit}$')
    else:
        ax.text(xy_text[0], xy_text[1], 
            f'$T_{{2e}}\\approx {tau:.2f}\\pm{tau_err:.4f} {unit}$')
    return cfit, ax

def fit_t2(logf, ax, xy_text, unit='\\mu s'):
    cfit = fitter.CurveFit(
        xdata=logf.df['delay_ns'].values,
        ydata=logf.df['s1_prob'].values,
        model=models.ExpSineModel(),
    )
    ax.plot(cfit.xdata, cfit.fdata(), 'r-', lw=1)
    tau = cfit.result.params["tau"].value
    tau_err = cfit.result.params["tau"].stderr
    ax.text(xy_text[0], xy_text[1], 
        f'$T_2^*\\approx {tau / 1e3:.2f}\\pm{tau_err / 1e3:.4f} {unit}$')
    return cfit, ax