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

def display_compare(temp,Dense):
    """
    Function to visualize matrix temp and its zoomed version in comparision with actual dense connectome.

    Returns:
    Just the plot.
    """
    zoom=3
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    
    im_1=axs[0].imshow(Dense, interpolation='nearest')
    fig.colorbar(im_1, ax=axs[0], orientation='vertical')
    axs[0].set_title('DC')
    axins = zoomed_inset_axes(axs[0], zoom, loc=1) # zoom = 6
    axins.imshow(Dense, interpolation="nearest",
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
  

    im_2=axs[1].imshow(temp, interpolation='nearest')
    fig.colorbar(im_2, ax=axs[1], orientation='vertical')
    axs[1].set_title('Approx. DC')

    axins = zoomed_inset_axes(axs[1], zoom, loc=1) # zoom = 6
    axins.imshow(temp, interpolation="nearest",
             origin="lower")
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area'''
    mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.draw()
    
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



