# Authors: Shun Chi (shunchi100@gmail.com)

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.exposure import match_histograms
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
import statistics
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from scipy.spatial import KDTree

def down_sample(data_array,factor,nn,coordinates):

    indices_picked=np.linspace(0, data_array.shape[0]-1, int(factor*(data_array.shape[0])), dtype=int)
    downsampled_matrix=data_array[indices_picked,:]
    
    # Step 1: Build a KD-Tree
    tree = KDTree(coordinates[indices_picked])
    
    _,closest_rows = tree.query(coordinates, k=nn)  # Nearest neighbor query

    return downsampled_matrix,closest_rows,indices_picked

def up_sample(downsampled_matrix,closest_rows):

    upsampled_matrix = downsampled_matrix[closest_rows]

    return upsampled_matrix


def create_cyclic_colormap(num_cycles, num_points=256, name='cyclic_colormap'):
    """
    Creates a Matplotlib colormap where the color (hue) cycles once, and brightness
    alternates between two values.

    Args:
        num_cycles (int): Number of brightness cycles.
        num_points (int): Number of points in the colormap.
        name (str): Name of the colormap.

    Returns:
        LinearSegmentedColormap: A Matplotlib colormap object.
    """
    # Generate normalized time values
    t = np.linspace(0, 1, num_points)
    
    # Hue cycles ONCE across the entire range
    hues = t  # Linear gradient from 0 to 1 for one complete hue cycle
    
    # Brightness modulation: square wave alternating between 0.5 and 1
    brightness = 0.7 + 0.3 * ((np.sin(2 * np.pi * num_cycles * t - np.pi / 2)) + 1)
    
    # Constant saturation
    saturation = np.ones_like(t)
    
    # Combine into HSV
    hsv = np.stack([hues, saturation, brightness], axis=1)
    
    # Convert HSV to RGB
    rgb = hsv_to_rgb(hsv)
    
    # Ensure RGB values are within valid range [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    # Create a Matplotlib colormap from the RGB values
    cmap = LinearSegmentedColormap.from_list(name, rgb)
    return cmap

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

def density_scatter(x,y,tx,ty,plot_baseline=False,plot_xy=False):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)#, edgecolor='')
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    if plot_baseline:
        ax.plot(x,np.zeros(x.shape),'--r')
    if plot_xy:
        ax.plot(x,x,'--r')
    #plt.show(block=False)
    return fig,ax

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def matrix_MIGP(C, n_dim=1000, d_pca=1000, keep_mean=True):
    """Apply incremental PCA to C
    Inputs:
    C (2D array) : should be wide i.e. nxN where N bigger than n
    We pretend that the matrix C is made of column blocks, each block is
    one 'subject', and 'time' is the column dimension.

    n_dim (int)  : C is split up into nXn_dim matrices
    n_pca (int)  : maximum number of pcs kept (set to n_dim if larger than n_dim)
    keep_mean (bool) : keep the mean of C

    Returns:
    reduced version of C (size nxmin(n_dim,n_pca)
    """
    # Random order for columns of C (create a view rather than copy the data)

    if keep_mean:
        C_mean = np.mean(C, axis=0, keepdims=True)
        print('mean shape: ',C_mean.shape)
        #raise(Exception('Not implemented keep_mean yet!'))

    if d_pca > n_dim:
        d_pca = n_dim

    print('...Starting MIGP')
    t = timer()
    t.tic()
    _, N = C.shape
    #random_idx = np.random.permutation(N)
    #Cview = C[:, random_idx]
    Cview = C.copy()
    Cview=demean(Cview)
    proj_mat=[]
    W = None
    for i in tqdm(range(0,N,n_dim)):
        data = Cview[:, i:min(i+n_dim, N+1)].T  # transpose to get time as 1st dimension
        if W is not None:
            W = np.concatenate((W, (data)), axis=0)
        else:
            W = (data)
        k = min(d_pca, n_dim)
        _, U  = eigsh(W@W.T, k)

        W = U.T@W
        proj_mat.append(U)
    data = W[:min(W.shape[0], d_pca), :].T

    print(f'...Old matrix size : {C.shape[0]}x{C.shape[1]}')
    print(f'...New matrix size : {data.shape[0]}x{data.shape[1]}')
    print(f'...MIGP done in {t.toc()} secs.')
    return data,proj_mat,C_mean
    
# demean a matrix
def demean(X, axis=0):
   # print(np.mean(X, axis=axis, keepdims=True).shape)
    return X - np.mean(X, axis=axis, keepdims=True)


# Helper class for timing
class timer:
    def __init__(self):
        """
        Matlab-style timer class
        t = timer()
        t.tic()
        .... do stuff
        t.toc()
        """
        self._t = time.time()
    def tic(self):
        self._t = time.time()
    def toc(self):
        return f'{time.time()-self._t:.2f}'

def inverse_MIGP(C, proj, mean, n_dim=1000, d_pca=1000):
    print('...Starting inverse_MIGP')
    t = timer()
    t.tic()
    Cview = C.copy()
    N=C.shape[0]
    W = None
    for i in tqdm(range(len(proj))):
        if W is None:
            data=Cview
            dim=proj[-1*(i+1)].shape[0]
            approx=data@proj[-1*(i+1)].T
          
            if len(proj)==1:
                W=(approx)
            else:
                W=(approx[:,d_pca:])
        else:
            data=approx[:,:d_pca]
            approx=data@proj[-1*(i+1)].T
            if i==(len(proj)-1):
                W=np.concatenate((approx,W), axis=1)
            else:
                W=np.concatenate((approx[:,d_pca:],W), axis=1)
      
    print(f'...Input matrix size : {C.shape[0]}x{C.shape[1]}')
    print(f'...New matrix size : {W.shape[0]}x{W.shape[1]}')
    print(f'...inverse_MIGP done in {t.toc()} secs.')
    return (W)+mean
        

def second_derivative(matrix):
    """
    Computes the second derivative of a given matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix.

    Returns:
    numpy.ndarray: The second derivative of the matrix.
    """
    # Ensure the input is a numpy array
    matrix = np.asarray(matrix)
    
    # Use np.gradient to compute the first derivative
    first_derivativex = np.gradient(matrix, axis=0)
    # Compute the second derivative from the first derivative
    second_derivativex = np.gradient(first_derivativex, axis=0)

    # Use np.gradient to compute the first derivative
    first_derivativey = np.gradient(matrix, axis=1)
    # Compute the second derivative from the first derivative
    second_derivativey = np.gradient(first_derivativey, axis=1)
    
    return np.sum(0.5*(np.abs(second_derivativex)+np.abs(second_derivativey)))

def intersect_masks(embedded,i):
    dx,dy=embedded.shape
    for j in range(dy):
        temp=np.zeros((dx,1))
        temp[np.abs(embedded[:,j]-embedded[i,j])<0.05*(np.max(embedded[:,j])-np.min(embedded[:,j]))]=1   
        temp[i]=2
        plt.imshow(np.reshape(temp,(Nv1,Nv2)))
        plt.show()

def intersect_visual(embedded):
    dx,dy=embedded.shape
    output=np.zeros((dx,1))
    for i in range(dx):
        mask=np.ones((dx,1))
        for j in range(dy):
            temp=np.zeros((dx,1))
            temp[np.abs(embedded[:,j]-embedded[i,j])<0.05*(np.max(embedded[:,j])-np.min(embedded[:,j]))]=1
            #plt.imshow(np.reshape(temp,(Nv1,Nv2)))
           # plt.show()
            mask*=temp
        output[i]=np.sum(mask)
    return output

def intersect_me(embedded):
    dx,dy=embedded.shape
    output=np.zeros((dx,1))
    for i in range(dx):
        mask=np.ones((dx,1))
        for j in range(dy):
            temp=np.zeros((dx,1))
            temp[np.abs(embedded[:,j]-embedded[i,j])<0.05*(np.max(embedded[:,j])-np.min(embedded[:,j]))]=1
            #plt.imshow(np.reshape(temp,(Nv1,Nv2)))
           # plt.show()
            mask*=temp
        output[i]=np.sum(mask)
    return np.percentile(output.flatten(),10)

def compute_distance_matrix(data):
    """
    Computes the pairwise distance matrix for an N x F data matrix.

    Parameters:
    data (numpy.ndarray): An N x F matrix where N is the number of points and F is the number of features.

    Returns:
    numpy.ndarray: An N x N distance matrix containing the pairwise distances among the N points.
    """
    # Compute the pairwise distances using pdist (which returns a condensed distance matrix)
    condensed_dist_matrix = pdist(data, metric='euclidean')
    
    # Convert the condensed distance matrix to a square distance matrix
    distance_matrix = squareform(condensed_dist_matrix)
    
    return distance_matrix.flatten()

def display_columns(matrix, Nv1, Nv2,type):
    """
    Function to visualize columns of a matrix in m separate plots in with shape of Nv1*Nv2 image.

    Parameters:
    matrix (numpy.ndarray): matrix of shape (Nv1*Nv2, m).
    Nv1 (integer): Width of the plotted image.
    Nv2 (integer): Height of the plotted image.

    Returns:
    Just the plot.
    """
    A, B = matrix.shape
    reshaped_columns = [np.reshape(matrix[:, i], (Nv1, Nv2)) for i in range(B)]
    
    fig, axs = plt.subplots(1, B, figsize=(B * 3, 3))  # Adjust figsize as needed
    for i, col in enumerate(reshaped_columns):
        axs[i].imshow(col, cmap=type)  # Change the colormap if needed
        axs[i].set_title(f'PC {i+1}')
        axs[i].axis('off')
    plt.show()

def display_compare(temp,Dense,text1='Approx. DC',text2='DC'):
    """
    Function to visualize matrix temp and its zoomed version in comparision with actual dense connectome.

    Returns:
    Just the plot.
    """
    zoom=3
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    
    im_1=axs[0].imshow(Dense)
    fig.colorbar(im_1, ax=axs[0], orientation='vertical')
    axs[0].set_title(text2)
    axins = zoomed_inset_axes(axs[0], zoom, loc=1) # zoom = 6
    axins.imshow(Dense,
             origin="lower")

    # sub region of the original image
    x1, x2, y1, y2 = 300, 100, 100, 300
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area'''
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.draw()
  

    im_2=axs[1].imshow(temp)
    fig.colorbar(im_2, ax=axs[1], orientation='vertical')
    axs[1].set_title(text1)

    axins = zoomed_inset_axes(axs[1], zoom, loc=1) # zoom = 6
    axins.imshow(temp,
             origin="lower")
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area'''
    mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.draw()
    plt.show()
    
def to_index(x, y, width):
    """
    to convert 2D indices to 1D index

    Parameters:
    x (integer): x coordinate of the point.
    y (integer): y coordinate of the point.
    width (integer): Height of the voxel slice.

    Returns:
    integer: The encoded index in 1D array.
    """
    return x * width + y

def create_graph(matrix):
    Nv1, Nv2 = matrix.shape
    num_nodes = Nv1 * Nv2
    indices = []
    indptr = [0]
    data = []

    for i in range(Nv1):
        for j in range(Nv2):
            if matrix[i, j] == 1:  # Only consider spiral points
                node_index = to_index(i, j, Nv2)
                neighbors = []
                if i > 0 and matrix[i-1, j] == 1:
                    neighbors.append(to_index(i-1, j, Nv2))
                if i < Nv1 - 1 and matrix[i+1, j] == 1:
                    neighbors.append(to_index(i+1, j, Nv2))
                if j > 0 and matrix[i, j-1] == 1:
                    neighbors.append(to_index(i, j-1, Nv2))
                if j < Nv2 - 1 and matrix[i, j+1] == 1:
                    neighbors.append(to_index(i, j+1, Nv2))

                indices.extend(neighbors)
                data.extend([1] * len(neighbors))
            indptr.append(len(indices))
    
    return csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    

def find_spiral_points(matrix):
    points = np.argwhere(matrix == 1)
    return points

def dist_vs_dist(mask,main_comp):
    """
    function to compare the distiance on a manifold (original space) with the distance in embedding space.

    Parameters:
    mask (numpy.ndarray): Manifold mask of shape (Nv1, Nv2)
    main_comp (numpy.ndarray): embedding of the points with the shape (Nv1*Nv2,1).

    Returns:
    Manifold_distance vs embedding distance correlation scatter plot.
    """    
    graph = create_graph(mask)
    Mss_points = find_spiral_points(mask)
    distance_manifold=[]
    distance_embedding=[]
    for i in range(Mss_points.shape[0]):
        point1 = Mss_points[i]
        for j in range(Mss_points.shape[0]):
            point2 = Mss_points[j]
            index1 = to_index(point1[0], point1[1], Nv2)
            index2 = to_index(point2[0], point2[1], Nv2)
            dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=index1, return_predecessors=True)
            distance_manifold .append(dist_matrix[index2])
            distance_embedding.append(np.abs(np.reshape(main_comp,(Nv1,Nv2))[Mss_points[i][0],Mss_points[i][1]]-np.reshape(main_comp,(Nv1,Nv2))[Mss_points[j][0],Mss_points[j][1]]))
    plt.scatter(distance_manifold,distance_embedding, alpha=0.2)
    plt.xlabel('Manifold_distance')
    plt.ylabel('Embedding_distance')
    print(f'Dist_to_Dist Correlation:{np.corrcoef(distance_manifold,distance_embedding)[0,1]}')

def dense_recon(embedded,ref_mat,low_rank,n):
    approx_dist=np.zeros_like(ref_mat)
    for i in range(n):
        approx_dist+=np.abs(np.subtract.outer(embedded[:,i],embedded[:,i]))**2
    approx_dist=(approx_dist)**0.5
    approx_dense=np.max(approx_dist)-approx_dist
    grad_dense=match_histograms(approx_dense, ref_mat)
    return low_rank+grad_dense
def visualize_evaluate_embedding(embedded,Lowrank_DC,Dense_C_res,Dense_C,colorcode,slice_size,Type="Manifold Learning"):
    M2=embedded.shape[1]; Nv1=slice_size[0]; Nv2=slice_size[1]
    display_columns(embedded, Nv1, Nv2,'viridis')
    display_columns(embedded, Nv1, Nv2,'prism')
    fig, axs = plt.subplots(1, 4, figsize=(13, 4))
    axs[0].scatter(embedded[:,0],embedded[:,1],c=colorcode.T, alpha=0.15, cmap='viridis')
    axs[0].set_title('Embedding Representation')
    axs[0].set_xlabel('1st Component')
    axs[0].set_ylabel('2nd Component')
    
    im = axs[1].imshow(np.reshape(colorcode.T,(Nv1,Nv2)),cmap='viridis')
    #im=axs[1].imshow(index_pattren,cmap='viridis')
    axs[1].set_title('Colormap')
    cbar = plt.colorbar(im, ax=axs[1])
    
    
    axs[2].scatter(embedded[:,0],embedded[:,1],c=intersect_visual(embedded), alpha=0.15, cmap='viridis')
    #axs[0].scatter(embedded[:,0],embedded[:,1],c=index_pattren.flatten(), alpha=0.15, cmap='viridis')
    axs[2].set_title('Embedding Representation')
    axs[2].set_xlabel('1st Component')
    axs[2].set_ylabel('2nd Component')
    
    
    im=axs[3].imshow(np.reshape(intersect_visual(embedded),(Nv1,Nv2)),cmap='viridis',vmin=0, vmax=200)
    axs[3].set_title('Localisation')
    cbar = plt.colorbar(im, ax=axs[3])
    plt.show()
    
    approx_dist=np.zeros((Nv1*Nv2,Nv1*Nv2))
    
    for i in range(M2):
        approx_dist+=np.abs(np.subtract.outer(embedded[:,i],embedded[:,i]))**2
    
    
    approx_dist=(approx_dist)**0.5
    
    approx_dense=(np.max(approx_dist)-approx_dist)
    
    temp1=match_histograms(approx_dense, Dense_C_res)
    
    print('Final Approximate DC:')
    approx_dense=temp1+Lowrank_DC

    display_compare(Lowrank_DC,Dense_C,text1='Low rank',text2='DC')
    display_compare(approx_dense,Dense_C)
    display_compare(temp1,Dense_C_res,text1='Approx. Grad',text2='DC_res')
    
    print(f'Quantitaive Full Corr Results for {Type}:')
    print('Correlation(low_rank_dense vs Dense_C):',np.corrcoef(Lowrank_DC.flatten(),Dense_C.flatten())[0,1])
    print('Correlation(grad_dense vs dense_res):',np.corrcoef(temp1.flatten(),Dense_C_res.flatten())[0,1])
    print('Correlation(approx_dense vs Dense_C):',np.corrcoef(approx_dense.flatten(),Dense_C.flatten())[0,1])
    print(f'Quantitaive Half Corr Results for {Type}:')
    print('Correlation(low_rank_dense vs Dense_C):',np.corrcoef(Lowrank_DC[np.triu_indices(Lowrank_DC.shape[0],k=1)],Dense_C[np.triu_indices(Dense_C.shape[0],k=1)])[0,1])
    print('Correlation(grad_dense vs dense_res):',np.corrcoef(temp1[np.triu_indices(temp1.shape[0],k=1)],Dense_C_res[np.triu_indices(Dense_C_res.shape[0],k=1)])[0,1])
    print('Correlation(approx_dense vs Dense_C):',np.corrcoef(approx_dense[np.triu_indices(approx_dense.shape[0],k=1)],Dense_C[np.triu_indices(Dense_C.shape[0],k=1)])[0,1])

    print(f'Quantitaive Full MSE Results for {Type}:')
    print('MSE (low_rank_dense vs Dense_C):',np.mean((Lowrank_DC - Dense_C) ** 2))
    print('MSE (grad_dense vs dense_res):',np.mean((Dense_C_res - temp1) ** 2))
    print('MSE (approx_dense vs Dense_C):',np.mean((Dense_C - approx_dense) ** 2))
    print(f'Quantitaive Half MSE Results for {Type}:')
    print('MSE (low_rank_dense vs Dense_C):',np.mean((Lowrank_DC[np.triu_indices(Lowrank_DC.shape[0],k=1)] - Dense_C[np.triu_indices(Lowrank_DC.shape[0],k=1)]) ** 2))
    print('MSE (grad_dense vs dense_res):',np.mean((Dense_C_res[np.triu_indices(Lowrank_DC.shape[0],k=1)] - temp1[np.triu_indices(Lowrank_DC.shape[0],k=1)]) ** 2))
    print('MSE (approx_dense vs Dense_C):',np.mean((Dense_C[np.triu_indices(Lowrank_DC.shape[0],k=1)] - approx_dense[np.triu_indices(Lowrank_DC.shape[0],k=1)]) ** 2))

    return temp1, approx_dense
    
'''def matrix_to_pmap_old(matrix, Nv1, Nv2, cycles):
    """
    Convert a matrix of scalar values into an RGB image representation.
    
    Parameters:
        matrix (numpy.ndarray): 2D array of scalar values.
        cycles (int): Number of periodic cycles for the red and green channels.
    
    Returns:
        numpy.ndarray: 3D RGB array representing the matrix visualization.
    """
    A, B = matrix.shape
    reshaped_columns = [np.reshape(matrix[:, i], (Nv1, Nv2)) for i in range(B)]
    fig, axs = plt.subplots(1, B+1, figsize=(B * 3, 3))  # Adjust figsize as needed
    for i, col in enumerate(reshaped_columns):
        # Normalize the matrix to 0-1 range
        normalized_matrix = (col- np.min(col)) / (np.max(col) - np.min(col))
    
        # Red and Green channels: Periodic mapping
        blue_channel = normalized_matrix
        green_channel = normalized_matrix 
    
        # Blue channel: Linear normalization
        red_channel = (normalized_matrix * cycles) % 1
    
        # Combine channels into an RGB image
        rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
        axs[i].imshow(rgb_image)  
        axs[i].set_title(f'PC {i+1}')
        axs[i].axis('off')
    temp=np.ones((100,10))
    for i in range(temp.shape[0]):
        temp[i,:]=(temp.shape[0]-i)/(temp.shape[0])
    normalized_matrix = (temp- np.min(temp)) / (np.max(temp) - np.min(temp))
    blue_channel = normalized_matrix
    green_channel = normalized_matrix
    red_channel = (normalized_matrix * cycles) % 1
    rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    axs[-1].imshow(rgb_image)  
    axs[-1].set_title(f'cmap')
    axs[-1].axis('off')
    plt.show()'''
def create_cyclic_hsv_colormap(cycles):
    """
    Creates a cyclic colormap based on the HSV colormap where each cycle reduces the brightness by 50%.

    Parameters:
    - cycles: int, the number of cycles for the colormap.

    Returns:
    - colormap: LinearSegmentedColormap, the custom cyclic colormap.
    """
    # Number of colors per cycle
    base_colors = 256  # Resolution of the colormap
    total_colors = cycles * base_colors  # Total colors for the colormap

    # Generate HSV values
    hues = np.linspace(0, 1, base_colors, endpoint=False)  # Hue values across the HSV colormap
    colormap_data = []

    for cycle in range(cycles):
        brightness = 0.5+(cycle/(2*cycles)) # Reduce brightness by 50% each cycle
        for h in hues:
            # Convert HSV to RGB
            r, g, b = plt.cm.hsv(h)[:3]  # Extract RGB from base HSV colormap
            colormap_data.append((brightness * r, brightness * g, brightness * b))

    # Convert to a custom colormap
    colormap = LinearSegmentedColormap.from_list(f'cyclic_hsv_{cycles}', colormap_data, N=total_colors)
    return colormap
    
def matrix_to_pmap(matrix, Nv1, Nv2, cycles):
    """
    Convert a matrix of scalar values into an RGB image representation.
    
    Parameters:
        matrix (numpy.ndarray): 2D array of scalar values.
        cycles (int): Number of periodic cycles for the red and green channels.
    
    Returns:
        numpy.ndarray: 3D RGB array representing the matrix visualization.
    """
    custom_cmap = create_cyclic_hsv_colormap(cycles)
    A, B = matrix.shape
    reshaped_columns = [np.reshape(matrix[:, i], (Nv1, Nv2)) for i in range(B)]
    fig, axs = plt.subplots(1, B, figsize=(B * 3, 3))  # Adjust figsize as needed
    for i, col in enumerate(reshaped_columns):
        im=axs[i].imshow(col,cmap=custom_cmap)  
        fig.colorbar(im, ax=axs[i], orientation='vertical')
        axs[i].set_title(f'PC {i+1}')
        axs[i].axis('off')
    plt.show()

def sample_rows(matrix, step):
    """
    Samples rows from the input matrix with a given step.
    
    Parameters:
        matrix (np.ndarray): The input 2D matrix.
        step (int): The step size for row sampling.
        
    Returns:
        np.ndarray: The sampled matrix.
    """
    return matrix[::step]

def inverse_sample(sampled_matrix, original_size, step):
    """
    Expands the sampled matrix to the original size by repeating rows.
    
    Parameters:
        sampled_matrix (np.ndarray): The sampled matrix.
        original_size (int): The number of rows in the original matrix.
        step (int): The step size used for sampling.
        
    Returns:
        np.ndarray: The expanded matrix with the original size.
    """
    repeated_rows = np.repeat(sampled_matrix, step, axis=0)
    return repeated_rows[:original_size]
