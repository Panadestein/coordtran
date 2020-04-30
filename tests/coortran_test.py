"""A series of unit tests for the coordtran module"""
import unittest
import numpy as np
import coordtran


class Testcoortran(unittest.TestCase):
    """Checks the correct behaviour of the functions
    in the coortran module by using some reference
    geometries"""

    def test_int_cart_int(self):
        """"Test that the convertion from internal
        coordinates to cartesians is a bijection
        """
        coords = np.array([-.3, 2., 2.1, 2.3, -0.2, -2.6, 1.19, 1.19, 2.34],
                          np.float64)
        int2cart = coordtran.coord_tran.int2cart
        cart2int = coordtran.coord_tran.cart2int
        self.assertEqual(cart2int(int2cart(coords)), coords)


if __name__ == "__main__":
    unittest.main()
