import numpy as np
import matplotlib.pyplot as plt
from labcodes.calc.base import Calculator, dept


class LC_MCoupled(Calculator):
    """LC resonator with mutual inductance to a transmission line.

    Finally equivalent to series RLC resonator. For details see examples & notes.
    """

    L = 20e-9
    C = 90e-15
    # Must be array with evenly spaced points.
    w = np.linspace(3e9, 5e9, 1001) * 2 * np.pi
    M = 10e-12
    R = 50
    Lf = 0e-12

    @dept
    def w00(self, L, C):
        return 1 / np.sqrt(L * C)

    @dept
    def zin(self, L, C, w, M, R, Lf):
        return (
            1 / (1j * w * C)
            + 1j * w * (L - M)
            + 1 / (1 / (1j * w * M) + 1 / (1j * w * (Lf - M) + 2 * R))
        )

    @dept
    def w0(self, zin, w):
        return np.interp(0, zin.imag, w)

    @dept
    def Lp(self, w0, w, zin):
        return np.interp(w0, w, np.gradient(zin.imag, w[1] - w[0])) / 2

    @dept
    def Rp(self, w0, w, zin):
        return np.interp(w0, w, zin.real)

    @dept
    def Q(self, w0, Lp, Rp):
        return w0 * Lp / Rp

    @dept
    def freq_shift(self, w0, w00):
        return (w0 - w00) / 2 / np.pi

    def demo_zin_vs_w(self):
        freq_GHz = self.w / 2 / np.pi / 1e9
        zin = self.zin()

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(4, 4))
        ax.plot(freq_GHz, zin.imag)
        ax2.plot(freq_GHz, np.gradient(zin.imag, self.w[1] - self.w[0]), "--")
        ax.set_ylabel("imag(zin)")
        ax2.set_ylabel("d/dw imag(zin)")
        ax2.set_xlabel("freq (GHz)")
        return fig, ax, ax2
    
    def demo_Q_vs_M(self, m_list=None):
        lc = self.copy()
        if m_list is None: m_list = np.linspace(10e-12, 100e-12, 10)
        lc_list = [LC_MCoupled(M=m) for m in m_list]
        Q_list = np.array([lc["Q"] for lc in lc_list])
        R_analytic = (lc["w00"] * m_list) ** 2 / (2 * lc["R"])
        Q_analytic = lc["w00"] * lc["L"] / R_analytic
        freq_shift_list = np.array([lc["freq_shift"] for lc in lc_list])

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(4,4))
        ax.plot(m_list/1e-12, Q_list/1e6)
        ax.plot(m_list/1e-12, Q_analytic/1e6, '--')
        ax2.plot(m_list/1e-12, freq_shift_list/1e6)
        ax2.plot(m_list/1e-12, np.zeros_like(m_list), '--')
        ax.set_ylabel("Q (million)")
        ax.set_yscale("log")
        ax2.set_ylabel("freq shift (MHz)")
        ax2.set_xlabel("M (pH)")
        return fig, ax, ax2

    def describe(self):
        w00 = self["w00"]
        w0 = self["w0"]
        Q = self["Q"]
        print(f"Unperturbed freq is {w00/(2*np.pi)/1e9:.3f} GHz")
        print(f"Q is {Q/1e6:.4f} million, freq shift is {(w0-w00)/1e6/2/np.pi:.3f} MHz")


class LC_CCoupled(Calculator):
    """LC resonator with capacitively coupled to a transmission line.

    Finally equivalent to parallel RLC resonator. For details see examples & notes.
    """

    L = 20e-9
    C = 90e-15
    # Must be array with evenly spaced points.
    w = np.linspace(3e9, 5e9, 1001) * 2 * np.pi
    Cg = 0.5e-15
    R = 50

    @dept
    def w00(self, L, C):
        return 1 / np.sqrt(L * C)

    @dept
    def zin_inv(self, w, L, C, Cg, R):
        return 1 / (1j * w * L) + 1j * w * C + 1 / (1 / (1j * w * Cg) + R)

    @dept
    def w0(self, zin_inv, w):
        return np.interp(0, zin_inv.imag, w)

    @dept
    def Cp(self, w0, w, zin_inv):
        return np.interp(w0, w, np.gradient(zin_inv.imag, w[1] - w[0])) / 2

    @dept
    def Rp(self, w0, w, zin_inv):
        return 1 / np.interp(w0, w, zin_inv.real)

    @dept
    def Q(self, w0, Rp, Cp):
        return w0 * Rp * Cp

    @dept
    def freq_shift(self, w0, w00):
        return (w0 - w00) / 2 / np.pi

    def demo_zin_vs_w(self):
        freq_GHz = self.w / 2 / np.pi / 1e9
        zin_inv = self.zin_inv()

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(4, 4))
        ax.plot(freq_GHz, zin_inv.imag)
        ax2.plot(freq_GHz, np.gradient(zin_inv.imag, self.w[1] - self.w[0]), "--")
        ax.set_ylabel("imag(zin$^{-1}$)")
        ax2.set_ylabel("d/dw imag(zin$^{-1}$)")
        ax2.set_xlabel("freq (GHz)")
        return fig, ax, ax2
    
    def demo_Q_vs_Cg(self, Cg_list=None):
        if Cg_list is None: Cg_list = np.linspace(0.1e-15, 1e-15, 10)
        lc_list = [LC_CCoupled(Cg=Cg) for Cg in Cg_list]
        Q_list = np.array([lc["Q"] for lc in lc_list])
        freq_shift_list = np.array([lc["freq_shift"] for lc in lc_list])

        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(4,4), sharex=True)
        ax.plot(Cg_list/1e-15, Q_list/1e6, label="Q")
        ax.set_ylabel("Q (million)")
        ax.set_yscale("log")
        ax2.plot(Cg_list/1e-15, freq_shift_list/1e6, '--', label="freq shift (MHz)")
        ax2.set_ylabel("freq shift (MHz)")
        ax2.set_xlabel("Cg (fF)")
        return fig, ax, ax2

    def describe(self):
        w00 = self["w00"]
        w0 = self["w0"]
        Q = self["Q"]
        print(f"Unperturbed freq is {w00/(2*np.pi)/1e9:.3f} GHz")
        print(f"Q is {Q/1e6:.4f} million, freq shift is {(w0-w00)/1e6/2/np.pi:.3f} MHz")
