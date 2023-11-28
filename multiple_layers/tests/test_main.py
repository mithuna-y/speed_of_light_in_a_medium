import unittest
import numpy as np
from main import electron_field_contribution, force, hookes_constant, damping_constant, DT, c, charge_electron  # Import your function

smallest_distance = c * DT

class TestElectronFieldContribution(unittest.TestCase):

    def test_zero_distance(self):
        # Arrange
        x_e, y_e, z_e = 1, 1, 1
        x_p, y_p, z_p = 1, 1, 1  # Same as electron position
        t_e, t_p = 0, 1
        accel_hist = np.array([0, 0, 0])

        # Act
        result = electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_hist)

        # Assert
        self.assertEqual(result, 0)


    def test_electron_contributing(self):
        # Arrange
        x_e, y_e, z_e, t_e = 0, 0, 0, 0
        x_p, y_p, z_p, t_p = smallest_distance, 0, 0, DT
        accel_hist = np.array([1, 2])

        # Act
        result = electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_hist)

        # Assert

        expected_result = charge_electron * (-1/smallest_distance)
        self.assertAlmostEqual(result, expected_result)

    def test_electron_not_contributing(self):
        # Arrange
        x_e, y_e, z_e, t_e = 0, 0, 0, DT
        x_p, y_p, z_p, t_p = smallest_distance, 0, 0, 3 * DT  # Time difference is too large
        accel_hist = np.array([1, 2, 3, 4])

        # Act
        result = electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_hist)

        # Assert
        self.assertEqual(result, 0)

    def test_zero_field(self):
        # Test with zero electric field
        total_electric_field = 0
        z = 1
        z_velocity = 1
        expected_force = -hookes_constant * z - damping_constant * z_velocity  # Based on your formula
        result = force(total_electric_field, z, z_velocity)
        self.assertAlmostEqual(result, expected_force)

    def test_no_movement(self):
        # Test with the electron at rest (no velocity and zero displacement)
        total_electric_field = 10
        z = 0
        z_velocity = 0
        expected_force = total_electric_field
        result = force(total_electric_field, z, z_velocity)
        self.assertAlmostEqual(result, expected_force)

    def test_normal_conditions(self):
        # Test under normal conditions
        total_electric_field = 10
        z = 5
        z_velocity = 2
        expected_force = total_electric_field - hookes_constant * z - damping_constant * z_velocity
        result = force(total_electric_field, z, z_velocity)
        self.assertAlmostEqual(result, expected_force)



if __name__ == '__main__':
    unittest.main()
