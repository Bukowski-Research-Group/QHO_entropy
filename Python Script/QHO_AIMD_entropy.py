"""
Brandon C. Bukowski
Mingze Zheng
Yanqi Huang
Geet Gupta
Johns Hopkins University
08/14/2024
Version 1.0
Citations:
Kabsch rotations: https://github.com/charnley/rmsd
VACF: Stefan Bringuier https://zenodo.org/records/5261318 DOI: 10.5281/zenodo.5261318
QHO: Alexopoulos, K. et al. 10.1021/acs.jpcc.6b00923
Application: Bukowski B.C. et al. 10.1016/j.jcat.2018.07.012 

v1.0 notes:
 - removed DOFs as an input since the DOFs are calculated from the VDOS integral, if you want to rescale the VDOS you can divide the FFT by this integral.
 - removed padding as an input since uniform padding was implimented and does not affect the VDOS integral.
 - removed tau as an input. Current implimentation does not account for VACF rescaling with different tau values. Currently disabled until fixed.
 - VACF is now unnormalized and has units of kg * m^2 / s^2 as expected. 
 - call_fft has been completely re-written to return the spectral density. the units are now kg * m^2 / s, which when divided by kT gives s^1 as expected.
 - added a new input called "length" which allows you to specify how long of a trajectory you want to calculate (i.e. your file is 100 timesteps but you only want to calculate 50)

Important notes - the input file should be pre-processed to only include production data on the adsorbate of interest
It is your responsibility to ensure the simulation is properly equilibrated - non-equilibrium structures will change the VDOS
and lead to larger than expect translational, rotational, and vibrational contributions to the VDOS

Remove any atoms that are not of interest for determining the entropy, i.e. surfaces, solids, solvents, etc. 
Each unique molecule type should be calculated separately

You must ensure that this trajectory does not include discontinuities around periodic boundaries
i.e. the trajectory should be "unwrapped". There are existing software that can perform this action.
"""


import argparse
import numpy as np
from ase.io import read, write
from ase.visualize import view
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.fftpack import fftfreq


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="integrate VDOS obtained from atomic trajectories",
        usage="Example, QHO_AIMD_entropy.py vasprun.xml 0.5e-15 500 --mode Vib",
    )

    parser.add_argument(
        "filename",
        help="the name of the trajectory file. Standard ASE filetypes are supported",
        type=str)

    parser.add_argument(
        "dt",
        type=float,
        help="timestep in units of s",
    )

    parser.add_argument(
        "temp",
        type=float,
        help="temperature in Kelvin",
    )

    parser.add_argument(
        "length",
        type=int,
        help="correlation length you want to calculate \
            this must be <= the length of the trajectory file \
            if not specified, this defaults to the total trajectory length",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="choose which DoF velocities to analyze \
            Total -- translation + rotation + vibration \
            Resid -- rotation + vibration \
            Vib   -- vibration \
            keep in mind rotation is ideal gas total rotation. \
            Recommended usage - for gas-phase molecules use 'Vib' to avoid double-counting rotational modes \
            - for adsorbate molecules use 'Resid' or 'Total' to use QHO approximation for all modes \
            unknown strings default to 'Total' and return an error",
    )


    args = parser.parse_args()

    return args 

def remove_COM_motion(images):
    """
    Takes a set of images, removes their COM motion, and returns the COM velocities
    unit conversion to true velocity is done in a later step
    
    Returns:
    modified set of images
    V_COM - center of mass velocities
    V_total - velocities of all atoms
    """
    len_molec = len(images[0]) # number of atoms
    
    # translate every image to the initial COM
    i_box = images[0].get_center_of_mass()
    for image in images:
        image.positions -= i_box  
        
    V_total = np.zeros((len(images),len_molec,3))
    V_COM = np.zeros((V_total.shape))

    for i in range(len(images)):
        if i == 0:
            V_total[i] = np.zeros((len_molec,3))
            V_COM[i] = np.zeros((len_molec,3))
        else:
            V_total[i,:,:] = images[i].positions - images[i-1].positions
            V_COM[i,:,:] = images[i].get_center_of_mass() - images[i-1].get_center_of_mass()

    V_total_sum = np.zeros((V_total.shape))
    V_COM_sum = np.zeros((V_COM.shape))
    
    for i in range(len(V_total[:,0,0])):
        V_total_sum[i] = np.sum(V_total[:i+1,:,:], axis=0)
        V_COM_sum[i] = np.sum(V_COM[:i+1,:,:], axis=0)

    for i in range(len(images[:])):
        images[i].positions -= V_COM_sum[i]
        
    return images, V_COM, V_total

def remove_rotational_motion(images):
    """
    Takes a set of images, removes their principal rotations, and returns the vibrational velocities
    
    Returns:
    modified set of images
    V_vib - vibrational velocities, no COM or rotational motion
    
    algorithm from: 
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    heavily inspired by Charnely RMSD package
    https://github.com/charnley/rmsd    
    """
    def kabsch(init, final):
        Q_C = init.positions.mean(axis=0)
        P_C = final.positions.mean(axis=0)
        Q = init.positions - Q_C
        P = final.positions - P_C

        H = np.dot(np.transpose(P), Q)    #  C
        U, S, V = np.linalg.svd(H)  # V S W
        d = np.linalg.det(U)*np.linalg.det(V) < 0.0
        if d:
            U[:,-1] = -U[:,-1]

        R = np.dot(U, V)

        P = np.dot(P, R)
        P += Q_C
        return P

    len_molec = len(images[0]) # number of atoms
    for image in images[1:]:
        image.positions = kabsch(images[0], image)

    V_vib = np.zeros((len(images),len_molec,3))

    for i in range(len(images)):
        if i == 0:
            V_vib[i] = np.zeros((len_molec,3))
        else:
            V_vib[i,:,:] = images[i].positions - images[i-1].positions
            
    return images, V_vib

def call_vacf(data, correlation_length, masses):
    """
    Takes a set of velocities, the VACF length, and the masses to return the VACF
    
    Returns:
    Normalized mass-weighted velocity autocorrelation function for a given set of velocities
    
    algorithm adapted from: 
    VACF: Stefan Bringuier https://zenodo.org/records/5261318 DOI: 10.5281/zenodo.5261318  
    """    
    def autocorr(X):
        out = np.correlate(X, X, mode='full')
        return out[out.size // 2:]
    
    len_images = len(data[:,0,0])
    total_VACF = np.zeros(correlation_length)
    
    blocks = range(correlation_length, len_images+1, correlation_length)
    
    for t in blocks:
        for j, ms in enumerate(masses):
            for i in range(3):
                total_VACF += autocorr(np.sqrt(ms) * data[t-correlation_length:t, j, i])
                #total_VACF += autocorr(ms * data[t-correlation_length:t, j, i])
    total_VACF /= len(blocks)
    time = np.arange(correlation_length)
    return time, total_VACF

def call_fft(vacf, dt, correlation):
    """
    Takes a VACF and returns the VDOS. Note this is set up to use a Hanning window
    There are many possible window functions in discrete signal processing that can be used
    It is worth taking a look at the trade-offs between peak accuracy and peak spillage 
    depending on your choice of window function. In our testing, this is a critically important
    factor in how the VDOS is calculated. 
    
    parameters:
    pad factor - how much zero-padding to use in the FFT
    dt - timestep in "s"
    correlation - the correlation length of the VACF
    
    Returns:
    unnormalized VDOS for a given VACF using a Hanning window function
    """ 
    vacf = vacf * np.hanning(vacf.size)
    og_nf = len(vacf)
    nfft = 25*og_nf # 25x zero pad - can be modified
    nfft2 = int(nfft/2)
    X = fft(vacf, nfft) /og_nf
    Xfreq = fftfreq(nfft, d=dt)
    Xfreq *= (1/299792458)/100 # convert from s^-1 to cm^-1 for plotting
    Xfreq = Xfreq[:nfft2]
    X = X[0:nfft2]
    X[1:nfft2] = 2*X[1:nfft2] # FFT symmetry
    
    return Xfreq, np.abs(X)

def integrate_VDOS(fft_normed, freq, T):
    """
    Takes a normalized VDOS, then integrates with a QHO weight function.
    
    parameters:
    T - temperature in K
    DoFs - How many DoFs to integrate, usually an integer between 3N-6 and 3N
      
    Returns:
    Entropy in J/mol/K
    
    Alexopoulos, K. et al. 10.1021/acs.jpcc.6b00923
    Bukowski B.C. et al. 10.1016/j.jcat.2018.07.012 
    """     
    def weight_function(freqs, T):
        hbar = 6.62607015E-34 # J/K
        kT = (1.380649E-23*T) # J
        x = hbar * freqs / (2*kT)
        return x/np.tanh(x) - np.log(2*np.sinh(x))

    freq_s = freq * 3e10 # convert back to s^-1 for integration
    set_weighted = fft_normed[1:] * weight_function(freq_s[1:], T)
    return np.trapz(set_weighted, x=freq[1:]) * 1 * 8.314 # R in J/mol/K


def main():
    args = parse_arguments()
    MD_filepath = args.filename
    vacf_mode = args.mode
    dt = args.dt 
    T = args.temp 
    length = args.length

    if vacf_mode == None:
        vacf_mode = "Total"
        print("Defaulting to using the total trajectories to make the VDOS")

    #len - 

    images = read(MD_filepath, index=f"-{length}:")
    #images = images[:length]
    print(len(images))

    # remove COM motion and calculate COM and total velocities
    images, V_COM, V_total = remove_COM_motion(images)
    # V_resid is the residual velocity after removing COM motion
    V_resid = V_total - V_COM
    # remove principal rotations to calculate vibrational velocity.
    # images are modified in-place
    images, V_vib = remove_rotational_motion(images)

    # clean this section up
    dt_au = dt*1e15 # convert from s to fs
    V_vib /= dt_au # divide by dt to get velocity in A/fs
    V_vib *= 1.0E5 # convert A/fs to m/s
    V_resid /= dt_au
    V_resid *= 1.0E5
    V_total /= dt_au
    V_total *= 1.0E5
    masses = images[0].get_masses()

    ncorr = len(images)
    masses = images[0].get_masses()

    if vacf_mode == "Vib":
        times, VACF = call_vacf(V_vib, ncorr, masses*1.66054e-27)
    elif vacf_mode == "Resid":
        times, VACF = call_vacf(V_resid, ncorr, masses*1.66054e-27)
    elif vacf_mode == "Total":
        times, VACF = call_vacf(V_total, ncorr, masses*1.66054e-27)
    else:
        times, VACF = call_vacf(V_total, ncorr, masses*1.66054e-27) 

    plt.plot(VACF)
    plt.title("VACF")
    plt.xlabel("correlation time / s")
    plt.ylabel("VACF / kg m^2 / s^2")
    plt.show()


    freq, ifft = call_fft(VACF, dt, ncorr)
    ifft *= dt #multiply fft bins by dt for spectral density

    kt = 1.380649E-23 * T
    ifft *= (2/kt) # normalize by KE
    ifft *= 2.998E10 # convert to cm so that x axis is cm-1 and y axis is cm
    ifft *= 1.63 # Hanning window correction factor from Discrete Signal Processing
    nDoF = np.trapz(ifft[:], x=freq[:]) # integral DOS which should equal nDoF
    entropy = integrate_VDOS(ifft, freq, T)
    print(f"Entropy at {T} K with {nDoF} degrees of freedom: {entropy} J Mol-1 K-1")


    plt.plot(freq, ifft)
    plt.title("VDOS")
    plt.xlim(0, 4500)
    plt.ylim(0, np.max(ifft))
    plt.xlabel("cm^-1")
    plt.ylabel("cm")
    plt.show()

    return None

if __name__ == "__main__":
    main()
