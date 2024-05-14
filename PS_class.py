import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants
from astropy import units as u
import scipy.interpolate
import math
import numpy as np

"""
_______________________________________________________________________________________________
"""

# Class for calculating the growth function
class GrowthFunction:
    def __init__(self, h=0.67, omega_m=0.32, omega_l=0.68):
        # Initializer with default cosmological parameters
        self.omega_m0 = omega_m
        self.omega_l0 = omega_l
        self.H_0 = ((h * 100 * u.km * u.s**-1 * u.Mpc**-1).to(u.Gyr**-1)).value
        self.h = h
    
    # Function to calculate cosmic time from redshift
    def Time(self, z):
        return ((2 * self.H_0**(-1)) / (1 + (1 + z)**2))

    # Function to calculate E(z), the evolution of H(z)=E(z)*H0
    def E_of_z(self, redshift):
        Ez = (self.omega_l0 + (1.0 - self.omega_l0 - self.omega_m0) * (1.0 + redshift)**2 + self.omega_m0 * (1.0 + redshift)**3)**(1/2)
        return Ez

    # Function to calculate matter density at redshift z
    def omega_mz(self, redshift):
        return self.omega_m0 * (1.0 + redshift)**3 / self.E_of_z(redshift)**2.0

    # Function to calculate vacuum density at redshift z
    def omega_lz(self, redshift):
        return self.omega_l0 / (self.E_of_z(redshift))**2

    # Function to calculate the linear growth factor
    def g_of_z(self, redshift):
        gz = 2.5 * self.omega_mz(redshift) / (self.omega_mz(redshift)**(4/7) - self.omega_lz(redshift) + ((1 + self.omega_mz(redshift)/2) * (1 + self.omega_lz(redshift)/70)))
        return gz

    # Function to calculate the growth function
    def D_of_z(self, redshift):
        return self.g_of_z(redshift) / self.g_of_z(0.0) / (1.0 + redshift)
    
    # Function to calculate the critical overdensity at redshift z
    def Delta_c(self, redshift):
        return 1.686 / self.D_of_z(redshift)


"""
_______________________________________________________________________________________________
"""


class Overdensities:
    
    def __init__(self, redshift=0, h7=0.962, h=0.67, omega_m=0.32, omega_l=0.68, ns=0.967, sigma8=0.808, Nbins_Sigma=50, logmass_lim=(4, 20)):

        # Validation of input parameters
        if (h <= 0.0):
            raise ValueError("Overdensities(): Negative Hubble constant illegal.\n")
        elif (h > 2.0):
            raise ValueError("Overdensities(): Reduced Hubble constant h should be in units of 100 km/s/Mpc.\n")
        if (h7 <= 0.0):
            raise ValueError("Overdensities(): Negative Hubble constant illegal.\n")
        elif (h7 > 2.0):
            raise ValueError("Overdensities(): Reduced Hubble constant h7 should be in units of 70 km/s/Mpc.\n")
        if (redshift <= -1.0):
            raise ValueError("Overdensities(): Redshift < -1 is illegal.\n")
        elif (redshift > 4000.0):
            raise ValueError("Overdensities(): Large redshift entered.  TF may be inaccurate.\n")

        # Initialize GrowthFunction object with given cosmological parameters
        Growth = GrowthFunction(omega_m=omega_m, omega_l=omega_l)

        # Assign input parameters to object attributes
        self.omega_m0 = omega_m
        self.omega_l0 = omega_l
        self.h7 = h7
        self.h = h
        self.Nbins_Sigma = Nbins_Sigma
        self.ns = ns
        self.redshift = redshift
        self.logmass_max = logmass_lim[1]
        self.logmass_min = logmass_lim[0]
        self.rho_0 = 133363631951.67577 # Mean matter density
        self.kpivot = 2e-3  # WMAP pivot scale

        # Assign growth function methods to object attributes
        self.D_of_z = Growth.D_of_z
        self.g_of_z = Growth.g_of_z

        # Normalization of the power spectrum
        self.N = 1
        self.N = sigma8 * sigma8 / (self.Sigma_of_R(11.4))**2

        # Output arrays
        self.R = np.empty(self.Nbins_Sigma, dtype='float64')  # Smoother radius [Mpc/h]
        self.M = np.empty(self.Nbins_Sigma, dtype='float64')  # mass (M_sun/h)
        self.logM = np.empty(self.Nbins_Sigma, dtype='float64')  # mass (log10 M_sun/h)
       
    """
    Main function 
    _______________________________________________________________________________________________
    """
    def S(self, m):
        # Vectorized version of S_prep
        s = np.vectorize(self.S_prep)
        return s(m)

    def S_Z(self, m, z):
        # Calculate S(m) at redshift z
        sz = self.D_of_z(z) * self.S(m)
        return sz
    """
    _______________________________________________________________________________________________
    """

    # Function to calculate the smoothing scale
    def S_prep(self, m):
        gamma_f, c = (2 * np.pi)**(2/3), 1.5
        R = c * (m / (gamma_f * self.rho_0))**(1/3)
        # Integrate the filter function
        s = integrate.quad(self.integrant, 0, np.inf, args=R, epsrel=1e-2, limit=100)[0]
        return (s / (2 * np.pi**2))**(1/2)

    # Transfer function
    def TransferFunction(self, k):
        q = 2.04 / (self.omega_m0 * self.h7**2)
        return np.log(1 + 2.34 * q * k) / (2.34 * q * k * (1 + 3.89 * q * k + (16.1 * q * k)**2 + (5.46 * q * k)**3 + (6.71 * q * k)**4)**(1/4))
    
    # Power spectrum
    def PowerSpectrum(self, k):
        return self.N * self.TransferFunction(k)**2 * k * (k / self.kpivot)**(self.ns - 1)
    
    """
    Now we define each filters. 
    Using the series expansion at low Rk make the integration more precise (for numerical integration).
    """

    # Top-Hat filter function
    def TopHatFilter(self, x):
        if x < 1e-2:
            # Use series expansion at low Rk for more precision
            return 1./3. - x**2/30. + x**4/840
        else:
            return 3 / (x)**3 * (np.sin(x) - x * np.cos(x))
    
    # Integrand function for calculating Sigma(R)
    def integrant(self, k, R):
        return k**2 * self.PowerSpectrum(k) * self.TopHatFilter(k * R)**2
    
    # Calculate Sigma(R)
    def Sigma_of_R(self, R):
        s2 = integrate.quad(self.integrant, 0, np.inf, args=(R), epsrel=1e-2, limit=100)[0]
        return (s2 / (2 * np.pi**2))**(1/2)


"""
_______________________________________________________________________________________________
"""

class HaloMassFunction:

    def __init__(self, redshift, omega_m=0.32, omega_l=0.68, h=0.67, ns=0.967, sigma8=0.808, mass_function=None, Nbins=50, logmass_lim=(6, 20)):
        # Initialize Overdensities object with given cosmological parameters
        self.overden = Overdensities(redshift, omega_m=omega_m, omega_l=omega_l, h=h, ns=ns, sigma8=sigma8, Nbins_Sigma=Nbins, logmass_lim=logmass_lim)
        # Assign logM and M attributes from Overdensities object
        self.logM = self.overden.logM
        self.M = self.overden.M

        # Assign mass function method based on input or default to press_schechter
        if mass_function == None:
            self.mass_function = self.press_schechter
        else:
            self.mass_function = mass_function.__get__(self)

        # Assign redshift and Delta_c attributes
        self.redshift = redshift
        Growth = GrowthFunction(omega_m=omega_m, omega_l=omega_l)
        self.Delta_c = Growth.Delta_c

        # Assign S attribute from Overdensities object
        self.S = self.overden.S

    """
    Main function 
    _______________________________________________________________________________________________
    """
    # Initial Mass Function (IMF) calculation
    def IMF(self, m, z):
        """Returns the halo mass function dn/dM in units of h^4 M_sun^-1 Mpc^-3
        Requires mass in units of M_sun /h """
        # Mean matter density at redshift z in units of h^2 Msolar/Mpc^3
        # This is rho_c in units of h^-1 M_sun (Mpc/h)^-3
        rho_0 = self.overden.rho_0 

        # Calculate derivative of log(sigma) and sigma
        dlogsigma, sigma = self.logderivative(m)
        # Calculate Press-Schechter mass function at redshift z
        mass_func = self.press_schechter_z(sigma, z)
        
        # Calculate and return the IMF
        IMF = np.abs(dlogsigma) * mass_func / m * rho_0
        return IMF
    """
    _______________________________________________________________________________________________
    """

    # Function to calculate derivative of log(sigma)
    def logderivative(self, M):
        sigma = self.S(M)
        sigma_plus = self.S(M**1.1)
        return (np.log(sigma_plus * sigma_plus) - np.log(sigma * sigma)) / np.log(0.1 * M), sigma
    
    # Press-Schechter mass function at given redshift z
    def press_schechter_z(self, sigma, z):
        """Press-Schechter (This form Lacey and Cole eq. 2.11 1993)"""
        nu = self.Delta_c(z) / sigma
        return np.sqrt(1/2 * np.pi) * nu * np.exp(-0.5 * nu * nu)

    # Press-Schechter mass function at redshift of the object
    def press_schechter(self, sigma):
        """Press-Schechter (This form Lacey and Cole eq. 2.11 1993)"""
        nu = self.Delta_c(self.redshift) / sigma
        return np.sqrt(1/2 * np.pi) * nu * np.exp(-0.5 * nu * nu)

    # Sheth-Tormen mass function
    def sheth_tormen(self, sigma):
        """Sheth-Tormen 1999, eq. 6"""
        nu = self.Delta_c(self.redshift) / sigma
        A = 0.3222
        a = 0.707
        p = 0.3
        return A * np.sqrt(2.0 * a / math.pi) * (1.0 + np.power(1.0 / (nu * nu * a), p)) * nu * np.exp(-0.5 * a * nu * nu)

    # Warren mass function
    def warren(self, sigma):
        """LANL fitting function - Warren et al. 2005, astro-ph/0506395, eqtn. 5 """
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982
        return A * (np.power(sigma, -1.0 * a) + b) * np.exp(-1.0 * c / sigma / sigma)

    # Watson FOF mass function
    def watson_FOF(self, sigma):
        """Watson 2012, eq. 12"""
        A = 0.282
        a = 2.163
        b = 1.406
        c = 1.210
        return A * (np.power(b / sigma, 1.0 * a) + 1) * np.exp(-1.0 * c / sigma / sigma)

class MergerRate:
    def __init__(self, redshift, redshift_lim=(0, 12), M_1=5e13, omega_m=0.27, omega_l=0.73, h=0.73, ns=0.95, sigma8=0.8, Nbins=50, logmass_lim=(6, 20)):
        # Initialize MergerRate object with given parameters
        self.omega_m0 = omega_m
        self.omega_l0 = omega_l
        self.logmass_lim = logmass_lim
        self.sigma8 = sigma8
        self.h = h
        self.Nbins = Nbins
        self.ns = ns
        self.redshift = redshift
        self.logmass_max = logmass_lim[1]
        self.logmass_min = logmass_lim[0]
        self.rho_0 = self.omega_m0 * 2.78e+11
        self.M_1 = M_1
        self.redshift_max_lim = redshift_lim[1]
        self.redshift_min_lim = redshift_lim[0]

        # Initialize HaloMassFunction object
        self.HMF = HaloMassFunction(redshift, omega_m=omega_m, omega_l=omega_l, h=h, ns=ns, sigma8=sigma8, Nbins=Nbins, logmass_lim=logmass_lim)
        
        # Initialize GrowthFunction object
        Growth = GrowthFunction(omega_m=omega_m, omega_l=omega_l)
        self.Delta_c = Growth.Delta_c
        self.Time = Growth.Time

        # Initialize Overdensities object for interpolation
        overden = Overdensities(0, omega_m=self.omega_m0, omega_l=self.omega_l0, h=self.h, ns=self.ns, sigma8=self.sigma8, Nbins_Sigma=self.Nbins, logmass_lim=self.logmass_lim)
        self.S = overden.S

    """Create a merger rate at a fixed final mass in terms of the redshift"""
    def MergerRate_M_Z(self, M2, M1, Z):
        # Calculate derivative of log(sigma) and sigma for M2
        der, sigma2 = self.logderivative(M2)
        # Calculate sigma for M1
        sigma1 = self.S(M1)

        # Calculate merger rate
        MR = M2 / (2 * np.pi)**(1 / 2) * np.abs(2 / 3 * self.Delta_c(Z) / self.Time(Z) * der) * ((sigma1 / sigma2)**2 / (sigma1**2 - sigma2**2))**(3 / 2) * np.exp(-0.5 * self.Delta_c(Z)**2 * (1 / sigma2**2 - 1 / sigma1**2))

        return MR
    """Gives the merger rate of the halo in term of the two masses M1 and M2 depending on z"""
    def MergerRate_M_Z2(self, M2, M1, Z):
        # Calculate derivative of log(sigma) and sigma for M2 using interpolation
        interpolate = self.overden.interpolates[1]
        der, sigma2 = self.derivative2(M2, interpolate)
        # Calculate sigma for M1 using interpolation
        sigma1 = interpolate(np.log10(M1))

        # Calculate merger rate
        MR = M2 / (2 * np.pi)**(1 / 2) * np.abs(2 / 3 * self.Delta_c(Z) / self.Time(Z) * der) * ((sigma1 / sigma2)**2 / (sigma1**2 - sigma2**2))**(3 / 2) * np.exp(-0.5 * self.Delta_c(Z)**2 * (1 / sigma2**2 - 1 / sigma1**2))

        return MR

    """Create a merger rate at a fixed redshift in terms of the mass"""
    def MergerRate_of_M2(self):
        # Initialize Overdensities object for interpolation
        overden = Overdensities(self.redshift, omega_m=self.omega_m0, omega_l=self.omega_l0, h=self.h, ns=self.ns, sigma8=self.sigma8, Nbins_Sigma=self.Nbins, logmass_lim=self.logmass_lim)
        interpolate = overden.interpolates[1]

        # Initialize arrays for mass range and merger rate
        DeltaM = np.empty(self.Nbins, dtype=object)
        LogdeltaM = np.empty(self.Nbins, dtype=object)
        MR = np.empty(self.Nbins, dtype=object)

        logDMmin = -2
        logDMmax = 2
        sigma1 = interpolate(np.log10(self.M_1))
        dlogm = (logDMmax - logDMmin) / self.Nbins

        # Loop over mass bins to calculate merger rate
        for i in range(self.Nbins):
            thislogDM = logDMmin + i * dlogm
            thisDM = 10**(thislogDM) * self.M_1
            thisM2 = thisDM + self.M_1

            der, sigma2 = self.logderivative2(thisM2, interpolate)

            MR[i] = 3 / 1e-3 * (2 / np.pi)**(1 / 2) / self.Time(self.redshift) * 2 / 3 * thisDM / thisM2 * np.abs(der) * self.Delta_c(self.redshift) / sigma2 / (1 - (sigma2 / sigma1)**2)**(3 / 2) * np.exp(-0.5 * self.Delta_c(self.redshift)**2 * (1 / sigma2**2 - 1 / sigma1**2))
            DeltaM[i] = thisDM
            LogdeltaM[i] = thislogDM

        return MR, DeltaM, LogdeltaM

    def logderivative(self, M):
        sigma = self.S(M)
        sigma_plus = self.S(1.00000000001 * M)
        return (sigma_plus * sigma_plus - sigma * sigma) / (0.00000000001 * M), sigma
            
    def logderivative2(self, M, interpolate):
        sigma = interpolate(np.log10(M))
        sigma_plus = interpolate(np.log10(1.1 * M))
        return (np.log(sigma_plus) - np.log(sigma)) / np.log(0.1 * M), sigma
    
    def derivative2(self, M, interpolate):
        sigma = interpolate(np.log10(M))
        sigma_plus = interpolate(np.log10(1.00000000001 * M))
        return (sigma_plus * sigma_plus - sigma * sigma) / (0.00000000001 * M), sigma
    
    def Delta_c_derivative(self, z):
        DT = self.Delta_c(z)
        DTplus = self.Delta_c(1.0001 * z)
        return (DTplus - DT) / 0.0001 * z
