# Authors: Shun Chi (shunchi100@gmail.com)

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.exposure import match_histograms
from scipy.stats import norm


def plot_mp_distribution_from_subset(eigenvalues,sig2,start,end,rank,N):
    c_best=1
    best_val=np.inf
    for c_test in np.linspace(0.001,100,10000):
        # Estimate lambda_min and lambda_max from subset of eigenvalues
        lambda_min = np.min([0.001,sig2*(1 - np.sqrt(c_test))**2])
        lambda_max = sig2*(1 + np.sqrt(c_test))**2
        # Generate the theoretical Marchenko–Pastur density over the estimated interval
        lambda_values = np.linspace(lambda_min, lambda_max, 1000)
        mp_density = (1 / (2 * np.pi *sig2* c_test * lambda_values)) * np.sqrt((lambda_max - lambda_values) * (lambda_values - lambda_min))
        # Normalize the density to get a CDF (cumulative sum normalized)
        mp_cdf = np.cumsum(mp_density)
        mp_cdf /= mp_cdf[-1]  # Normalize to make it a valid CDF
        # Create an interpolation function for the inverse CDF
        inverse_cdf = interp1d(mp_cdf, lambda_values, bounds_error=False, fill_value=(lambda_min, lambda_max))
        draw=np.zeros_like(eigenvalues)
        uniform_samples = np.random.rand(rank)
        mp_samples= inverse_cdf(uniform_samples)
        ascending_indices = mp_samples.argsort() 
        descending_indices = ascending_indices[::-1] 
        draw[:rank] = mp_samples[descending_indices]
        if np.linalg.norm((eigenvalues[start:end]-draw[start:end]))<best_val:
            c_best=c_test
            best_val=np.linalg.norm((eigenvalues[start:end]-draw[start:end]))
            
    # Estimate lambda_min and lambda_max from subset of eigenvalues
    lambda_min = np.min([0.001,sig2*(1 - np.sqrt(c_best))**2])
    lambda_max = sig2*(1 + np.sqrt(c_best))**2
    print(f'Lambda min: {lambda_min}, Lambda max: {lambda_max}')
    
    
    # Generate the theoretical Marchenko–Pastur density over the estimated interval
    lambda_values = np.linspace(lambda_min, lambda_max, 1000)
    mp_density = (1 / (2 * np.pi *sig2* c_best * lambda_values)) * np.sqrt((lambda_max - lambda_values) * (lambda_values - lambda_min))
    # Normalize the density to get a CDF (cumulative sum normalized)
    mp_cdf = np.cumsum(mp_density)
    mp_cdf /= mp_cdf[-1]  # Normalize to make it a valid CDF
    # Create an interpolation function for the inverse CDF
    inverse_cdf = interp1d(mp_cdf, lambda_values, bounds_error=False, fill_value=(lambda_min, lambda_max))

    mp_samples_array = np.zeros((N, eigenvalues.shape[0]))
    for i in range(N):
        # Generate uniform random samples and map them through the inverse CDF
        uniform_samples = np.random.rand(rank)
        mp_samples= inverse_cdf(uniform_samples)
        ascending_indices = mp_samples.argsort() 
        descending_indices = ascending_indices[::-1] 
        mp_samples_array[i, :rank] = mp_samples[descending_indices]

    # Calculate mean and standard deviation across the repetitions
    mp_mean = np.mean(mp_samples_array, axis=0)
    mp_std = np.std(mp_samples_array, axis=0)

    
    # Plot the original eigenvalue spectrum
    plt.plot(eigenvalues, '.', label='Eigenvalues Spectrum')

    # Plot the mean of generated MP samples with variance band
    plt.plot(mp_mean, '-', color='red', label=f'Mean MP (sorted c={c_best:.2f})')
    plt.fill_between(np.arange(eigenvalues.shape[0]), mp_mean - 2*mp_std, mp_mean + 2*mp_std, color='red', alpha=0.3, label='Variance Band (±2 std)')
    # Labeling the plot
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Spectrum")
    plt.legend()
    plt.grid(True)
    plt.show()    

    # Step 6: Plot histogram of empirical eigenvalues and overlay MP density
    plt.figure(figsize=(10, 6))

    # Plot histogram of eigenvalues
    plt.hist(eigenvalues, bins=100, density=True, alpha=0.5, color='skyblue', label='Empirical Eigenvalues')

    # Plot the MP density curve
    plt.plot(lambda_values, mp_density, 'r-', lw=2, label=f'Theoretical MP, c={c_best:.2f}')
    # Labeling the plot
    plt.xlabel("Eigenvalue")
    plt.ylabel("Pr value")
    plt.title("Marchenko–Pastur Density Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

def density_scatter(x,y,plot_baseline=False,plot_xy=False):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)#, edgecolor='')
    if plot_baseline:
        ax.plot(x,np.zeros(x.shape),'--r')
    if plot_xy:
        ax.plot(x,x,'--r')
    #plt.show(block=False)
    return fig,ax
