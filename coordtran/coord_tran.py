"""
Functions to convert Cartesian coordinates to internal coordinates
of the Pyr-NO2 system. The functions are in principle
applicable to any triatomic molecule over a fix surface.

The reference Pyr geometry have been optimized at wB97x/cc-pvDZ level and its
centered and aligned to the canonical Cartesian frame. The Pyr molecule is
considerend frozed in its equilibrium geometry and in the YZ plane, with its
principal axis aligned with the canonical ones.

"""
import numpy as np
from scipy.spatial.transform import Rotation as rotor

# Import reference pyrene (rotate to align our system)

GEOPYR = np.loadtxt("./pyreneref")

# Atom types and masses (nist.gov)

ATOMS = {1: ["H", 1.007], 6: ["C", 12.0096],
         7: ["N", 14.006], 8: ["O", 15.999]}

# Generate array with atomic masses needed for int2cart (practical)

MASNOO = np.array([elem[1] for elem in map(ATOMS.get, [7, 8, 8])],
                  dtype=np.float64)

# Function to generate cartesian coordinates


def int2cart(inter_coord):
    """
    Transforms internal coordinates to cartesian for the Pyr-NO2 system.

    The internal coordinates are contained in an array of shape (N, 9),
    with N being the number of geometries:

    `inter_coord` = [`cent_rel`, `alpha`, `beta`, `gamma`, `r1`, `r2`, `phi`]

    Parameters
    ----------

    `cent_rel` : array
        Coordinates of the center of mass of NO2 with respect to
        the origin (0.0, 0.0, 0.0)

    `alpha`, `beta`, `gamma` : float
        Euler angles of the NO2 molecular frame with respect
        to the reference canonical frame

    `r_1`, `r_2`, `phi` : float
        Internal coordinates of the NO2 molecule (distances and angle)

    Returns
    -------

    `cartesians` : array
        Array of shape (N, 4) containing the cartesian coordinates

    Notes
    -----

    The starting point geometry of NO2 is defined in the YZ plane (as the Pyr)
    With the N atom laying in the Z axis, and the first oxigen in the negative
    side of the Y axis. The center of mass is located in the origin

    All distances should be reported in Angstrom and angles in radians

    """
    # Unpack input array

    *cent_rel, alpha, beta, gamma, r_1, r_2, phi = inter_coord

    # Generate the NO2 on top of the Pyr (YZ plane)

    geonoo_0 = np.array([[0.0, 0.0, 0.0],
                         [0.0, -r_1 * np.cos(phi / 2), -r_1 * np.sin(phi / 2)],
                         [0.0, -r_2 * np.cos(phi / 2), r_2 * np.sin(phi / 2)]],
                        dtype=np.float64)

    # Translate NO2 to its center of mass (smart numpy broadcasting)

    geonoo_0 -= np.sum(geonoo_0 * MASNOO[:, None], axis=0) / np.sum(MASNOO)

    # Rotate and displace NO2 according to cent_rel and euler angles

    eulrot = rotor.from_euler('XYZ', np.array([alpha, beta, gamma]))
    rotmat = np.around(eulrot.as_matrix(), decimals=5)

    geonoo = np.transpose(rotmat @ geonoo_0.T)

    if not __debug__:
        # Print matrices and geometries to debug
        print("=====INT2CART======")
        print("geo NO2")
        print(geonoo)
        print("Mass NO2")
        print(MASNOO)
        print("Itens NO2")
        print(intens(MASNOO, geonoo))
        print("Axes NO2 (Eigvects of itens)")
        print(paxes(intens(MASNOO, geonoo)))
        print("Rotmat")
        print(rotmat)
        print(np.linalg.det(rotmat))
        print("===================")

    geonoo += cent_rel

    # Generate NO2 geometry with atomic numbers included

    anum = np.array([7, 8, 8])
    geonoo_mass = np.insert(geonoo, 0, anum, axis=1)

    # Return new cartesian geometry

    return np.around(np.concatenate((GEOPYR, geonoo_mass), axis=0),
                     decimals=5)


# Function to generate internal coordinates from Cartesian


def cart2int(cart_coord):
    """
    Transforms Cartesian coordinates to internal for the Pyr-NO2 system.

    The Cartesian coordinates are contained in an array of shape (N, 4),
    with N being the number of atoms and the first column corresponding
    to the atomic number.

    Parameters
    ----------

    `cart_coord` : array
        Array containing the Cartesian coordinates.

    Returns
    -------

    `internals` : array
        Array of shape (9,) containing the internal coordinates

    Notes
    -----

    When the principal axis of rotation are computed by diagonalization of
    an inertia tensor, an extra step is needed to get the correct direction
    of them. This is due to the fact that eigenvalues are defined up to a
    multiplicative constant. After the automatic normalization of them by
    Numpy, an annoying minus sign may appear. To fix this, a convention is
    set with the NO2 geometry generated in `int2cart`. This convention is
    verified here with scalar products.
    """

    # Separate molecular fragments (coord, atom type and map to generate mass)

    geopyr = cart_coord[cart_coord[:, 0] < 7.0][:, 1:]  # Filter Z < 14
    atomspyr = cart_coord[cart_coord[:, 0] < 7.0][:, 0].astype(int)
    maspyr = np.array([elem[1] for elem in map(ATOMS.get, atomspyr)],
                      dtype=np.float64)

    geonoo = cart_coord[cart_coord[:, 0] > 6.0][:, 1:]
    atomsnoo = cart_coord[cart_coord[:, 0] > 6.0][:, 0].astype(int)
    masnoo = np.array([elem[1] for elem in map(ATOMS.get, atomsnoo)],
                      dtype=np.float64)

    # Generate NO2 internal coordinates (assumes N as first atom)

    r_1 = np.linalg.norm(geonoo[0] - geonoo[1])
    r_2 = np.linalg.norm(geonoo[0] - geonoo[2])
    phi = np.arccos(np.dot(geonoo[0] - geonoo[1],
                           geonoo[0] - geonoo[2]) / (r_1 * r_2))

    # Compute fragments centers of mass

    cent_pyr = np.sum(geopyr * maspyr[:, None], axis=0) / np.sum(maspyr)

    # Translate both geometries to the pyrene frame

    geopyr -= cent_pyr
    geonoo -= cent_pyr

    # Compute principal axes of pyrene

    axes_pyr = paxes(intens(maspyr, geopyr))

    # Rotate both molecules to the pyrene frame (symbolic in the Pyr case)

    geonoo = np.transpose(axes_pyr.T @ geonoo.T)

    # Compute NO2 center of mass in new frame, and then translate to center
    # This center of mass is the distance between pyr and NO2

    cent_noo = np.sum(geonoo * masnoo[:, None], axis=0) / np.sum(masnoo)
    geonoo -= cent_noo

    # Compute NO2 principal axis

    axes_noo = paxes(intens(masnoo, geonoo))

    # Make sure the orientation of the NO2 is equivalent to int2cart's

    if np.dot((geonoo[2] - geonoo[0]), axes_noo[:, 1]) > 0.0:
        axes_noo[:, 1] *= -1
    if np.dot((geonoo[2] - geonoo[1]), axes_noo[:, 2]) < 0.0:
        axes_noo[:, 2] *= -1
    axes_noo[:, 0] = np.cross(axes_noo[:, 1], axes_noo[:, 2])

    # The rotation matrix should be equivalent to the NO2 axes

    rotmat = axes_noo

    if not __debug__:
        # Print atrices and geometries to debug
        print("======CART2INT=========")
        print("geo NO2")
        print(geonoo)
        print("Mass NO2")
        print(masnoo)
        print("Itens NO2")
        print(intens(masnoo, geonoo))
        print("Axes NO2 (Eigvects of itens)")
        print(axes_noo)
        print("Rotmat")
        print(rotmat)
        print(np.linalg.det(rotmat))
        print("=======================")

    rotacion = rotor.from_matrix(rotmat)
    alpha, beta, gamma = rotacion.as_euler('XYZ')

    # Return internal coordinates

    return np.around(np.array([*cent_noo, alpha, beta, gamma, r_1, r_2, phi],
                              dtype=np.float64), decimals=5)


# Function to get the moment of inertia tensor of a molecule


def intens(masses, coords):
    """
    Calculates the moment of inertia of a molecule by computing
    each element independently. Vectorial approaches may be better
    suited for large systems.

    Parameters
    ----------

    masses :  array
        Contains the atomic masses for each atom

    coords : array
        Array of shape (N, 3) containing the cartesian coordinates
        of the molecule.

    Returns
    -------

    itensor : array
        Array of shape (3, 3) containing the inertia tensor

    """
    # Define place holders

    inercia = np.zeros((3, 3), np.float64)

    # Compute elements of inertia tensor

    for mass, [x_c, y_c, z_c] in zip(masses, coords):
        inercia[0, 0] += mass * (y_c ** 2 + z_c ** 2)
        inercia[1, 1] += mass * (x_c ** 2 + z_c ** 2)
        inercia[2, 2] += mass * (x_c ** 2 + y_c ** 2)
        inercia[0, 1] += -mass * x_c * y_c
        inercia[0, 2] += -mass * x_c * z_c
        inercia[1, 2] += -mass * y_c * z_c
    inercia[1, 0] = inercia[0, 1]
    inercia[2, 0] = inercia[0, 2]
    inercia[2, 1] = inercia[1, 2]

    return np.around(inercia, decimals=5)


# Function to obatain the principal axes of a molecule


def paxes(itens):
    """
    Diagonalizes an inertia tensor to obtain the (sorted) principal
    axes of the molecule. Basically a wrapper to numpy's `eigh`

    Parameters
    ----------

    itens : array
        Array of shape (3, 3) containing the elements of the inertia tensor

    Returns
    -------

    priaxes : array
        Array whose columns contain the principal axesi
        sorted by largest eigenvalue.
    """

    # Diagonalize inertia tensor

    eigvals, eigvecs = np.linalg.eigh(itens)

    # Principal axis of the molecule

    order = np.argsort(eigvals)

    return eigvecs[:, order][:, ::-1]


# Function to write geometries to .xyz file


def wrt_geo(geom, name):
    """
    Writes a Pyr-NO2 cartesian geometry to a file.

    Parameters
    ----------

    geom: array(N, 9)
        The cartesian coordinates of the system
    name: str
        The name given to the file
    """
    with open(f"./{name}", "w") as outf:
        outf.write(f"{len(geom)}\n")
        outf.write("\n")
        for cart in geom:
            for idf, elem in enumerate(cart):
                if idf == 0:
                    outf.write(f"{ATOMS[int(elem)][0]} ")
                else:
                    outf.write(str(elem) + " ")
            outf.write("\n")


# Test geometry

if __name__ == "__main__":
    #  COORDS = np.array([-.3, 2., 2.1, 2.3, -0.2, -2.6, 1.19, 1.19, 2.34],
    #                    np.float64)
    #  print(COORDS)
    #  print(cart2int(int2cart(COORDS)))
    GEO = np.loadtxt('./SPs/minf_ts18')
    print(cart2int(GEO))
    print(cart2int(int2cart(cart2int(GEO))))
