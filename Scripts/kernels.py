"""Functions to generate kernels for fitting raster models to simulation data."""

import numpy as np
from scipy import special

def make_exponential_kernel(beta, scale):

    def exponential_kernel(distance):
        """Calculate exponential kernel."""
        return beta * np.exp(-distance / scale) / (2 * np.pi * scale * scale)

    return exponential_kernel

def make_exponential_jac(beta, scale):

    def exponential_jac(distance):
        """Calculate exponential kernel jacobian."""
        jac = np.array([
            np.exp(-distance / scale) / (2 * np.pi * scale * scale),
            beta * ((distance/(2*scale)) - 1) * np.exp(-distance / scale) / (
                np.pi * scale * scale * scale)
        ])

        return jac

    return exponential_jac

def make_cauchy_kernel(beta, scale):

    def cauchy_kernel(distance):
        """Calculate Cauchy kernel."""
        return 2 * beta / (scale * np.pi * (1 + np.power(distance / scale, 2)))

    return cauchy_kernel

def make_cauchy_jac(beta, scale):

    def cauchy_jac(distance):
        """Calculate exponential power kernel jacobian."""
        jac = np.array([
            2 / (scale * np.pi * (1 + np.power(distance / scale, 2))),
            2 * beta * ((2*distance*distance/(scale*scale + distance*distance)) - 1) / (
                scale * scale * np.pi * (1 + np.power(distance / scale, 2)))
        ])

        return jac

    return cauchy_jac

def make_exp_power_kernel(beta, power, scale):

    def exp_power_kernel(distance):
        """Calculate exponential power kernel."""
        return beta * power * np.exp(-np.power(distance / scale, power)) / (
            2 * np.pi * scale * scale * special.gamma(2.0/power))

    return exp_power_kernel

def make_exp_power_jac(beta, power, scale):

    def exp_power_jac(distance):
        """Calculate exponential power kernel jacobian."""
        jac = np.array([
            power * np.exp(-np.power(distance / scale, power)) / (
                2 * np.pi * scale * scale * special.gamma(2.0/power)),
            beta  * np.exp(-np.power(distance / scale, power)) * (
                1 - np.nan_to_num(power*np.power(distance / scale, power)*np.log(distance / scale))+
                2*special.psi(2.0/power)/power) / (
                    2 * np.pi * scale * scale * special.gamma(2.0/power)),
            beta * power * np.exp(-np.power(distance / scale, power)) * (
                power * np.power(distance / scale, power) / 2 - 1) / (
                    np.pi * scale * scale * scale * special.gamma(2.0 / power))
        ])

        return jac

    return exp_power_jac
