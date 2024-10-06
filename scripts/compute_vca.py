import argparse
import numpy as np
from numpy.typing import ArrayLike
from sentinel2_ts.utils.load_model import load_data
import matplotlib.pyplot as plt

def successive_projection_algorithm(data: ArrayLike, number_endmember: int):
    """
    Compute pure endmembers as well as coordinates of these pure endmembers using successive projection algorithm

    Args:
        data (ArrayLike): data from which endmembers are extracted
        number_endmember (int): number of endmembers to extract

    Returns:
        Arraylike: specters of the extracted endmembers 
    """

    data_to_project = data # initialize projected data
    endmember_coordinates = [] # initialize list of endmember indices

    for _ in range(number_endmember):
        norms = np.linalg.norm(data_to_project, axis=0)
        k = np.argmax(norms)
        u = data_to_project[:,k]/np.linalg.norm(data_to_project[:,k])
        data_to_project = data_to_project - np.dot(np.outer(u,u),data_to_project)
        endmember_coordinates.append(k)
        
    endmembers = data[:, endmember_coordinates]
    
    return(endmembers, endmember_coordinates)

def create_argparse():

    parser = argparse.ArgumentParser(description="Script for extracting abundances using VCA")

    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument("--number_endmember", type=int, default=3, help="Number of endmembers to extract")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the extracted endmembers")
    parser.add_argument("--scale_data", type=bool, default=True, help="Scale data")
    parser.add_argument("--clipping", type=bool, default=True, help="Clipping the data or not")
    return parser

def main():
    args = create_argparse().parse_args()

    data = load_data(args)[0]
    nb_band, x_range, y_range = data.shape
    data = data.reshape(nb_band, -1).T
    endmembers, endmember_coordinates = successive_projection_algorithm(data, args.number_endmember)
    np.save(args.save_path, endmembers)
    coordinates = np.unravel_index(endmember_coordinates,(x_range, y_range))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(data.reshape(nb_band, x_range, y_range)[[2, 1, 0]], cmap='viridis')
    ax.scatter(coordinates[1], coordinates[0], c='r', s=100)
    
if __name__ == "__main__":
    main()