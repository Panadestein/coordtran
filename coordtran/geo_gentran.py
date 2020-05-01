"""Transforms the reference Pyr-NO2 geometries to internal"""
import os
import io
import numpy as np
import coord_tran

# Initialize variables

GAUS = True
ATOMS = {1: ["H", 1.007], 6: ["C", 12.0096],
         7: ["N", 14.006], 8: ["O", 15.999]}
INTERNALS = []
BODYGAUS = """%mem=3GB\n%NProc=12
#P wB97XD/cc-pVDZ Integral(SuperFineGrid)

Single point calculation

0 2
"""

# Define function to write xyz file


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


# Define function to write gaussian inputs


def gausgen(geometria, fname):
    """
    Generates a Gaussian input file for
    the provided cartesian geometry of Pyr-NO2.
    The level of theory is hard coded here.
    Since we deal with Numpy arrays here,
    we have to buffer the array in a BytesIO object
    and then obtain the corresponding string.
    """
    bytesfil = io.BytesIO()
    np.savetxt(bytesfil, geometria,
               fmt=["%d", "%10.5f", "%10.5f", "%10.5f"])
    elarray = bytesfil.getvalue().decode('latin1')

    tofile = f"""%chk={fname}\n{BODYGAUS}{elarray} """

    with open(f"GAUSSINPS/{fname}.com", "w") as fout:
        fout.write(tofile)


# Define function to generate MOPAC inputs


def mopacgen(geometria, fname):
    """Generates a simple mopac input file from Cartesian coordinates"""
    bytesfil = io.BytesIO()
    np.savetxt(bytesfil, geometria,
               fmt=["%d", "%10.5f", "%10.5f", "%10.5f"])
    elarray = bytesfil.getvalue().decode('latin1')

    tofile = f"""pm7 1scf\nNonesen title\n\n{elarray} """

    with open(f"MOPACINPS/{fname}.mop", "w") as fout:
        fout.write(tofile)


# Main loop loading reference geometries


for filref in os.listdir('SPs/'):
    geocart = np.loadtxt(f"SPs/{filref}")
    newgeoint = coord_tran.cart2int(geocart)
    INTERNALS.append(newgeoint)
    np.savetxt(f"NEWSPs/{filref}_new",
               np.around(coord_tran.int2cart(newgeoint), decimals=5),
               fmt=["%d", "%10.5f", "%10.5f", "%10.5f"])

    if not __debug__:
        print(filref)
        print(newgeoint)
        print(coord_tran.cart2int(coord_tran.int2cart(newgeoint)))

    if GAUS:
        gausgen(geocart, filref)
        gausgen(coord_tran.int2cart(newgeoint), f"{filref}_new")
        mopacgen(geocart, filref)
        mopacgen(coord_tran.int2cart(newgeoint), f"{filref}_new")

    wrt_geo(geocart, f"XYZs/{filref}.xyz")
    wrt_geo(coord_tran.int2cart(newgeoint), f"XYZs/{filref}_new.xyz")

# Save internal coordinates

np.savetxt('intcoords', np.array(INTERNALS), fmt="%10.5f")
