# -*- coding: utf-8 -*-
"""
Created on Thu May  4 08:41:42 2023

@author: shileiee
"""
import numpy as np
import pandas as pd
import scipy.fftpack as fftpack
import scipy.linalg as linalg
from scipy.signal import chirp, get_window
from enum import Enum
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import pandas as pd

class AngleEstimationAlgorithm(Enum):
    DBF = 1
    AngleFFT = 2
    MUSIC = 3
    OMP = 4
    DML = 5

class RadarSimulator:
    c = 3e8  # Speed of light in meters per second
    def __init__(self, params):
        self.params = params
        self.data = None
        self.tx_antenna_coordinates = self.params["tx_antenna_coordinates"]
        self.rx_antenna_coordinates = self.params["rx_antenna_coordinates"]


    def generate_chirp_waveform(self):
        """
        1. Generate the transmit waveform.
        2. Calculate the echo delay and Doppler frequency shift based on the input target distance, angle, and velocity.
        3. Generate the echo signal for each channel in the antenna array.
        4. Load the antenna radiation pattern from an Excel file (if provided, otherwise use default values).
        5. Generate the received data matrix based on the antenna layout and multiple target parameters.
        """
        # 1. Generate the transmit waveform
        self.tx_waveform = self.generate_tx_waveform()

        # 2. Calculate the echo delay and Doppler frequency shift
        self.echo_delays, self.doppler_shifts = self.calculate_echo_params()

        # 3. Generate the echo signal for each channel in the antenna array
        self.echo_signals = self.generate_echo_signals()

        # 4. Load the antenna radiation pattern from an Excel file (if provided, otherwise use default values)
        self.antenna_gains = self.load_antenna_pattern()

        # 5. Generate the received data matrix
        self.rx_data_matrix = self.generate_rx_data_matrix()

    def generate_tx_waveform(self):
        """
        Generate the transmit waveform based on the radar parameters.
        """
        T_chirp = self.params["chirp_duration"]  # Chirp duration
        B = self.params["chirp_bandwidth"]  # Chirp bandwidth
        fs = self.params["sampling_frequency"]  # Sampling frequency
        num_samples = self.params["num_samples"]  # Number of samples
        t = np.linspace(0, T_chirp, num_samples, endpoint=False)  # Time array

        # Generate fast chirp waveform
        tx_waveform = chirp(t, f0=self.params["center_frequency"] - B / 2, f1=self.params["center_frequency"] + B / 2, t1=T_chirp, method='linear')

        return tx_waveform

    def calculate_echo_params(self):
        """
        Calculate the echo delay and Doppler frequency shift based on the input target distance, angle, and velocity.
        """
        target_distances = self.params["target_distances"]
        target_angles = self.params["target_angles"]
        target_velocities = self.params["target_velocities"]


        # Calculate echo delays
        echo_delays = [2 * distance / self.c for distance in target_distances]

        # Calculate Doppler frequency shifts
        doppler_shifts = [-2 * self.params["frequency"] * velocity / self.c for velocity in target_velocities]

        return echo_delays, doppler_shifts

    def load_antenna_pattern(self, antenna_pattern_file=None):
        """
        Load antenna pattern from an Excel file. If no file is provided, use a default value.
        """
        if antenna_pattern_file:
            antenna_pattern_df = pd.read_excel(antenna_pattern_file)
            # Assuming the antenna pattern data has two columns: 'angle' and 'gain'
            angles = antenna_pattern_df['angle'].values
            gains = antenna_pattern_df['gain'].values
            return dict(zip(angles, gains))
        else:
            # Use a default antenna pattern (e.g., isotropic or omnidirectional)
            return {angle: 1 for angle in range(-180, 181)}

    def generate_echo_signals(self, tx_waveform, antenna_pattern):
        """
        Generate echo signals for each channel, considering target parameters and antenna array layout.
        """
        echo_signals = []

        # Iterate through each target
        for i, (distance, angle, velocity) in enumerate(zip(self.params["target_distances"], self.params["target_angles"], self.params["target_velocities"])):
            # Calculate echo signal delay and Doppler frequency shift for each target
            delay = 2 * distance / self.c
            doppler_shift = 2 * velocity / (self.c / self.params["center_frequency"])

            # Iterate through each receive antenna
            for rx_idx in range(self.params["num_rx_antennas"]):
                # Apply antenna gain based on target angle
                antenna_gain = antenna_pattern.get(angle, 1)

                # Generate echo signal for each receive antenna considering delay and Doppler shift
                time_shifted_waveform = np.roll(tx_waveform, int(delay * self.params["sampling_frequency"]))
                freq_shifted_waveform = np.exp(-1j * 2 * np.pi * doppler_shift * np.arange(time_shifted_waveform.size) / self.params["sampling_frequency"]) * time_shifted_waveform

                # Apply antenna gain and add echo signal to the list
                echo_signals.append(antenna_gain * freq_shifted_waveform)

        # Combine echo signals for each channel
        echo_signals = np.array(echo_signals).reshape(self.params["num_rx_antennas"], -1)

        return echo_signals   

    def generate_rx_data_matrix(self, echo_signals):
        """
        Combine echo signals for each channel into a receive data matrix.
        """
        return np.stack(echo_signals, axis=0)

    def apply_windowing(self, data_matrix, window_function=None):
        """
        Apply a window function to the receive data matrix.
        """
        if window_function is None:
            # Default to no window (rectangular window)
            window_function = np.ones_like(data_matrix[0, :])

        # Apply the window function to each channel
        return data_matrix * window_function[np.newaxis, :]

    def add_noise(self, data_matrix, noise_level):
        """
        Add noise to the receive data matrix.
        """
        signal_power = np.mean(np.abs(data_matrix)**2)
        noise_power = signal_power / (10**(noise_level / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.normal(size=data_matrix.shape) + 1j * np.random.normal(size=data_matrix.shape))

        return data_matrix + noise

    def process_1D_FFT(self, data_matrix):
        """
        Perform 1D FFT on the receive data matrix to get range profile.
        """
        return np.fft.fftshift(np.fft.fft(data_matrix, axis=1), axes=1)

    def process_2D_FFT(self, range_data_matrix):
        """
        Perform 2D FFT on the range data matrix to get range-Doppler profile.
        """
        return np.fft.fftshift(np.fft.fft(range_data_matrix, axis=0), axes=0)

    def extract_targets_from_2DFFT(self, range_doppler_data):
        """
        Extract multi-channel complex data at the target locations from the range-Doppler profile.
        """
        # Find the target indices based on the input target parameters
        target_indices = []
        for target in self.params["targets"]:
            target_range, target_angle, target_speed = target
            range_idx = np.argmin(np.abs(self.params["range_bins"] - target_range))
            doppler_idx = np.argmin(np.abs(self.params["doppler_bins"] - target_speed))
            target_indices.append((range_idx, doppler_idx))

        # Extract multi-channel complex data at the target locations
        targets_data = []
        unique_target_indices = list(set(target_indices))
        for unique_idx in unique_target_indices:
            range_idx, doppler_idx = unique_idx
            target_data = range_doppler_data[:, range_idx, doppler_idx]

            # Handle multiple targets with the same range and Doppler indices
            if target_indices.count(unique_idx) > 1:
                targets_data.extend([target_data] * target_indices.count(unique_idx))
            else:
                targets_data.append(target_data)

        return np.array(targets_data)

    def process_MIMO(self):
        """
        Implement MIMO processing with a 1D horizontal array.
        """
        num_tx = len(self.tx_antenna_coordinates)
        num_rx = len(self.rx_antenna_coordinates)
        
        # Calculate MIMO virtual array coordinates
        self.virtual_array_coordinates = np.array([tx + rx for tx in self.tx_antenna_coordinates for rx in self.rx_antenna_coordinates])

        # Initialize MIMO processed data as a copy of target extracted 2D FFT data
        self.mimo_processed_data = self.target_extracted_data.copy()

        return self.virtual_array_coordinates


    def angle_estimation(self, algorithm, mimo_data):
        """
        Implement different angle estimation algorithms: DBF, Angle FFT, MUSIC, OMP, DML.
        """
        if algorithm == "DBF":
            return self.angle_estimation_DBF(mimo_data)
        elif algorithm == "Angle_FFT":
            return self.angle_estimation_AngleFFT(mimo_data)
        elif algorithm == "MUSIC":
            return self.angle_estimation_MUSIC(mimo_data)
        elif algorithm == "OMP":
            return self.angle_estimation_OMP(mimo_data)
        elif algorithm == "DML":
            return self.angle_estimation_DML(mimo_data)
        else:
            raise ValueError("Invalid angle estimation algorithm specified")

    def steering_matrix(self, angles, is_tx=True):
        """
        Calculate the steering matrix for a given set of angles.
        """
        if is_tx:
            antenna_coordinates = self.params["tx_antenna_coordinates"]
        else:
            antenna_coordinates = self.params["rx_antenna_coordinates"]
        wavelength = self.c / self.center_frequency

        response_matrix = np.exp(1j * 2 * np.pi / wavelength * np.outer(np.sum(antenna_coordinates * np.array([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))]), axis=1), np.ones_like(angles)))
        return response_matrix

    def angle_estimation_DBF(self, mimo_data):
        # Implement Digital Beamforming (DBF) algorithm
        steering_vectors = self.compute_steering_vectors(self.virtual_array_coordinates, self.params["angles_range"])
        angle_spectrum = np.abs(np.dot(steering_vectors.T, mimo_data))**2
        return angle_spectrum
    
    def angle_estimation_AngleFFT(self, mimo_data):
        # Implement Angle FFT algorithm
        N_fft = self.params["angle_fft_size"]
        angle_range = self.params["angles_range"]
        steering_vectors = self.compute_steering_vectors(self.virtual_array_coordinates, angle_range)
        
        fft_data = np.fft.fftshift(np.fft.fft(mimo_data, n=N_fft, axis=1), axes=1)
        angle_spectrum = np.abs(np.dot(steering_vectors.T, fft_data))**2
        return angle_spectrum
    
    def angle_estimation_MUSIC(self, mimo_data):
        # Implement MUSIC (Multiple Signal Classification) algorithm
        R = np.dot(mimo_data, mimo_data.conj().T)
        eig_values, eig_vectors = np.linalg.eig(R)
        idx = eig_values.argsort()[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
    
        noise_subspace = eig_vectors[:, self.params["num_targets"]:]
        angle_range = self.params["angles_range"]
        steering_vectors = self.compute_steering_vectors(self.virtual_array_coordinates, angle_range)
        
        denominator = np.abs(np.dot(steering_vectors.conj().T, noise_subspace))**2
        angle_spectrum = 1 / np.sum(denominator, axis=1)
        return angle_spectrum
    
    def angle_estimation_OMP(self, mimo_data):
        # Implement Orthogonal Matching Pursuit (OMP) algorithm
        angle_range = self.params["angles_range"]
        steering_vectors = self.compute_steering_vectors(self.virtual_array_coordinates, angle_range)
        
        residual = mimo_data
        approximated_data = np.zeros_like(mimo_data)
        
        for _ in range(self.params["num_targets"]):
            correlations = np.abs(np.dot(steering_vectors.conj().T, residual))
            max_idx = np.argmax(correlations)
            approximated_data += steering_vectors[:, max_idx].reshape(-1, 1) * np.dot(steering_vectors[:, max_idx].conj().T, residual)
            residual = mimo_data - approximated_data
    
        angle_spectrum = np.abs(approximated_data)**2
        return angle_spectrum
    
    def angle_estimation_DML(self, mimo_data):
        # Implement Deterministic Maximum Likelihood (DML) algorithm
        num_targets = len(self.params["target_params"])
        num_virtual_antennas = len(self.virtual_array_coordinates)
    
        # Calculate covariance matrix
        R = np.cov(mimo_data.T)
    
        # Calculate steering matrix
        steering_matrix = np.zeros((num_virtual_antennas, self.params["angle_resolution"]), dtype=complex)
        for i, angle in enumerate(np.linspace(-90, 90, self.params["angle_resolution"])):
            steering_vector = self.calculate_steering_vector(angle, self.virtual_array_coordinates)
            steering_matrix[:, i] = steering_vector
    
        # Calculate J matrix
        J = np.zeros((num_targets, self.params["angle_resolution"]), dtype=complex)
        for i in range(num_targets):
            R_inv = np.linalg.inv(R[i * num_virtual_antennas:(i + 1) * num_virtual_antennas,
                                    i * num_virtual_antennas:(i + 1) * num_virtual_antennas])
            J[i, :] = np.diag(steering_matrix.T.conj() @ R_inv @ steering_matrix)
    
        # Calculate angle spectrum
        angle_spectrum = np.zeros(self.params["angle_resolution"])
        for i in range(self.params["angle_resolution"]):
            angle_spectrum[i] = np.real(np.prod(J[:, i]))
    
        return angle_spectrum

    def evaluate_antenna_layout(self):
        """
        Evaluate different antenna layouts based on the given input parameters.
        """
        pass

    def visualize(self, angle_spectrum):
        # Plot echo signals, 1D FFT spectrum and 2D FFT spectrum
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot echo signals
        axs[0].plot(self.echo_signals[0, :])
        axs[0].set_title("Echo Signals")
        axs[0].set_xlabel("Time Samples")
        axs[0].set_ylabel("Amplitude")
        
        # Plot 1D FFT spectrum
        axs[1].plot(np.abs(self.range_data_matrix[0, :]))
        axs[1].set_title("1D FFT Spectrum")
        axs[1].set_xlabel("Range Bins")
        axs[1].set_ylabel("Magnitude")
        
        # Plot 2D FFT spectrum
        im = axs[2].imshow(np.abs(self.range_doppler_data[:, :, 0]), aspect="auto", origin="lower")
        axs[2].set_title("2D FFT Spectrum")
        axs[2].set_xlabel("Doppler Bins")
        axs[2].set_ylabel("Range Bins")
        fig.colorbar(im, ax=axs[2])
        
        plt.tight_layout()
        plt.show()
        
        # Plot angle estimation results for different algorithms
        fig, ax = plt.subplots(figsize=(12, 4))
        
        for algo_name, spectrum in angle_spectrum.items():
            ax.plot(self.angle_grid, spectrum, label=algo_name)
        
        ax.set_title("Angle Estimation Results")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Spectrum")
        ax.legend()
        plt.show()
        
        # Plot antenna array layout and equivalent MIMO array
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.scatter(self.params["tx_antenna_coordinates"][:, 0], self.params["tx_antenna_coordinates"][:, 1], marker="x", label="Transmit Antennas")
        ax.scatter(self.params["rx_antenna_coordinates"][:, 0], self.params["rx_antenna_coordinates"][:, 1], marker="o", label="Receive Antennas")
        ax.scatter(self.virtual_array_coordinates[:, 0], self.virtual_array_coordinates[:, 1], marker="d", label="Virtual MIMO Array")
        
        ax.set_title("Antenna Array Layout")
        ax.set_xlabel("X (wavelength)")
        ax.set_ylabel("Y (wavelength)")
        ax.legend()
        plt.show()

def simulate(self):
    """
    Simulate the whole radar processing chain.
    """
    # Generate the transmitted waveform
    tx_waveform = self.generate_tx_waveform()

    # Load the antenna pattern
    antenna_pattern = self.load_antenna_pattern(self.params["antenna_pattern_file"])

    # Generate echo signals
    echo_signals = self.generate_echo_signals(tx_waveform, antenna_pattern)

    # Generate the receive data matrix
    rx_data_matrix = self.generate_rx_data_matrix(echo_signals)

    # Apply windowing
    windowed_data_matrix = self.apply_windowing(rx_data_matrix, self.params["window_function"])

    # Add noise
    noisy_data_matrix = self.add_noise(windowed_data_matrix, self.params["noise_level"])

    # Process 1D FFT
    range_data_matrix = self.process_1D_FFT(noisy_data_matrix)

    # Process 2D FFT
    range_doppler_data = self.process_2D_FFT(range_data_matrix)

    # Extract targets from 2D FFT
    targets_data = self.extract_targets_from_2DFFT(range_doppler_data)

    # Process MIMO
    self.virtual_array_coordinates = self.process_MIMO(targets_data)

    angle_spectrum = self.angle_estimation(self.params["angle_estimation_algorithm"])
    self.evaluate_antenna_layout()
    self.visualize(angle_spectrum)

def main():
    # Define radar parameters
    params = {
        "center_frequency": 77e9,
        "chirp_duration": 1e-5,
        "chirp_bandwidth": 500e6,
        "sampling_frequency": 1e9,
        "num_samples": 1024,
        "num_tx_antennas": 4,
        "num_rx_antennas": 4,
        "angle_estimation_algorithm": AngleEstimationAlgorithm.MUSIC,
        "window_function": "hamming",
        "noise_level": -30,
        "target_distances": [100, 200],
        "target_angles": [30, -30],
        "target_velocities": [30, -30],
         "tx_antenna_coordinates": np.array([
        [0.0, 0],
        [0.5, 0],
        [1.2, 0],
        [1.8, 0]
                ]),  # 示例发射天线坐标，可根据需要自定义
        "rx_antenna_coordinates": np.array([
        [2.5, 0],
        [3.0, 0],
        [3.7, 0],
        [4.3, 0]
                ]),  # 示例接收天线坐标，可根据需要自定义
    # ...
        # Other parameters ...
    }

    # Initialize radar simulator
    radar_simulator = RadarSimulator(params)

    # Perform the simulation
    radar_simulator.simulate()

if __name__ == "__main__":
    main()
    
    

