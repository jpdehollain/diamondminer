from numpy import ndarray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import scipy as sp 
from typing import Optional

@dataclass
class CoulombDiamond:
    """Class for keeping track of Coulomb diamond information."""
    name: str
    left_vertex: tuple
    top_vertex: tuple
    right_vertex: tuple
    bottom_vertex: tuple
    oxide_thickness: float = None # m
    epsR: float = None # F/m
    e: float = 1.60217663e-19 # C
    eps0: float = 8.8541878128e-12 # F/m

    def width(self) -> float:
        return np.abs(self.right_vertex[0] - self.left_vertex[0])
    
    def height(self) -> float:
        return np.abs(self.top_vertex[1] - self.bottom_vertex[1])
    
    def beta(self) -> float:
        # positive slopes (drain lever arm)
        beta1 = (self.top_vertex[1] - self.left_vertex[1]) / (self.top_vertex[0] - self.left_vertex[0])
        beta2 = (self.bottom_vertex[1] - self.right_vertex[1]) / (self.bottom_vertex[0] - self.right_vertex[0])

        return 0.5 * (beta1 + beta2)
    
    def gamma(self) -> float:
        # positive slopes (source lever arm)
        gamma1 = -1 * (self.top_vertex[1] - self.right_vertex[1]) / (self.top_vertex[0] - self.right_vertex[0])
        gamma2 = -1 * (self.bottom_vertex[1] - self.left_vertex[1]) / (self.bottom_vertex[0] - self.left_vertex[0])
        return 0.5 * (gamma1 + gamma2)
    
    def alpha(self) -> float:
        return (self.beta()**-1 + self.gamma()**-1)**-1

    def lever_arm(self) -> float:
        return self.charging_voltage() / self.addition_voltage() 
    
    def addition_voltage(self) -> float:
        return self.width()
    
    def charging_voltage(self) -> float:
        return self.height() / 2

    def total_capacitance(self) -> float:
        return self.e / self.charging_voltage()
    
    def source_capacitance(self) -> float:
        return self.gate_capacitance() / self.gamma()

    def drain_capacitance(self) -> float:
        return self.gate_capacitance() * ((1- self.beta()) / self.beta() )
    
    def gate_capacitance(self) -> float:
        return self.e / self.addition_voltage()
    
    def dot_area(self) -> float:
        # Assumes parallel plate capicitaor geometry
        dot_area = (self.oxide_thickness * self.total_capacitance()) / (self.eps0 * self.epsR)
        return dot_area

    def dot_radius(self) -> float:
        if self.oxide_thickness is None:
            # Less accurate
            if self.epsR is None:
                return -1
            #https://arxiv.org/pdf/1910.05841
            return self.total_capacitance() / (8 * self.eps0 * self.epsR)
        else:
            #https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-05700-9/MediaObjects/41467_2018_5700_MOESM1_ESM.pdf
            dot_radius = np.sqrt(self.dot_area() / np.pi)
            return dot_radius

    def print_summary(self):
        print(f"Summary ({self.name}):")
        print("===================")
        print("\n")

        print("Constants")
        print("---------")
        print(f"Elementary Charge (e): {self.e:.5e} C")
        print(f"Permittivity of Free Space (\u03F50): {self.eps0:.5e} F/m")
        if self.epsR is not None:
            print(f"Relative Permittivity (\u03F5R): {self.epsR:.5f}")
        if self.oxide_thickness is not None:
            print(f"Oxide Thickness: {self.oxide_thickness * 1e9:.5f} nm")
        print("---------")
        print("\n")

        print("Geometry")
        print("---------")
        print(f"Left Vertex: {self.left_vertex}")
        print(f"Top Vertex: {self.top_vertex}")
        print(f"Right Vertex: {self.right_vertex}")
        print(f"Bottom Vertex: {self.bottom_vertex}")
        print(f"Width: {self.width()*1e3:.5f} mV")
        print(f"Height: {self.height()*1e3:.5f} mV")
        print("---------")
        print("\n")
        
        print("Dot Properties")
        print("--------------")
        print(f"Total Lever Arm (\u03B1): {self.lever_arm():.5f} eV/V")
        print(f"Drain Lever Arm (\u03B2): {self.beta():.5f} eV/V")
        print(f"Source Lever Arm (\u03B3): {self.gamma():.5f} eV/V")
        print(f"Addition Voltage: {self.addition_voltage() * 1e3:.5f} mV")
        print(f"Charging Voltage: {self.charging_voltage()* 1e3:.5f} mV")
        print(f"Gate Capacitance: {self.gate_capacitance() * 1e18:.5f} aF")
        print(f"Source Capacitance: {self.source_capacitance() * 1e18:.5f} aF")
        print(f"Drain Capacitance: {self.drain_capacitance() * 1e18:.5f} aF")
        print(f"Total Capacitance: {self.total_capacitance() * 1e18:.5f} aF")
        print(f"Dot Radius: {self.dot_radius() * 1e9:.5f} nm")
        print("--------------")
        print("\n")

    def plot(self, ax):
        vertices = [self.left_vertex, self.top_vertex, self.right_vertex, self.bottom_vertex, self.left_vertex]
        polygon = patches.Polygon(vertices, closed=True, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(polygon)
        
        # Calculate the center of the diamond for the label
        center_x = (self.left_vertex[0] + self.right_vertex[0]) / 2
        ax.text(center_x, 0, self.name, color='blue', ha='center', va='center', fontsize=10, weight='bold')

class Miner:

    def __init__(self,
                gate_data: ndarray,
                ohmic_data: ndarray,
                current_data: ndarray,
                epsR: Optional[float] = None,
                oxide_thickness: Optional[float] = None,
                binary_threshold: float = 1.1,
                binary_threshold_linear: float = None,
                blur_sigma: float = 1.0,
                blur_kernel: tuple = (3,3)) -> None:
        self.epsR = epsR
        self.oxide_thickness = oxide_thickness
        self.gate_data = gate_data
        self.ohmic_data = ohmic_data
        self.current_data = current_data

        self.current_data_height, self.current_data_width = self.current_data.shape
        self.image_ratio = self.current_data_height / self.current_data_width
        self.gate_voltage_per_pixel = np.abs((self.gate_data[-1] - self.gate_data[0]) / self.current_data_width)
        self.ohmic_voltage_per_pixel = np.abs((self.ohmic_data[-1] - self.ohmic_data[0]) / self.current_data_height)

        self.binary_threshold = binary_threshold
        self.binary_threshold_linear = binary_threshold_linear
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
    
    def filter_raw_data(self, 
                        current_data: ndarray,
                        binary_threshold: float = 1.1,
                        binary_threshold_linear: float = None,
                        blur_sigma: float = 1.0,
                        blur_kernel: tuple = (3,3)) -> ndarray:
        assert binary_threshold >= 1

        if binary_threshold_linear is not None:
            mask = current_data < binary_threshold_linear 
            new_data = np.zeros_like(current_data)

        else:
            filtered_current_data = np.log(
                np.abs(current_data)
            )            
            mask = filtered_current_data < binary_threshold * np.nanmax(filtered_current_data)
            new_data = np.zeros_like(filtered_current_data)


        new_data[mask] = 255
        new_data = new_data.astype(np.uint8)

        if blur_sigma < 0:
            return new_data

        # Apply Gaussian blur to smooth the image and reduce noise
        filtered_data = cv2.GaussianBlur(
            new_data, 
            blur_kernel, 
            blur_sigma
        )
        
        return filtered_data

    def extract_diamonds(self, debug: bool = False) -> None:

        upper_mask = np.zeros_like(self.current_data, dtype=bool)
        upper_mask[:self.current_data_height//2, :] = True

        lower_mask = np.zeros_like(self.current_data, dtype=bool)
        lower_mask[self.current_data_height//2:, :] = True

        masks = {
            'upper': upper_mask,
            "lower": lower_mask
        }

        line_dict = {
            'upper': {'positive': [], 'negative': []},
            'lower': {'positive': [], 'negative': []}
        }

        if debug:
            plt.title("Filtered Data + Detected Lines")
            plt.imshow(
                self.filter_raw_data(
                    self.current_data,
                    binary_threshold=self.binary_threshold,
                    blur_sigma=self.blur_sigma,
                    blur_kernel=self.blur_kernel
                ),
                cmap='binary', 
                aspect='auto'
            )
            plt.show()

        for section in ["upper", "lower"]:
            image = self.current_data * masks[section]
            image_threshold = self.filter_raw_data(
                image,
                binary_threshold=self.binary_threshold,
                blur_sigma=self.blur_sigma,
                blur_kernel=self.blur_kernel
            )
            image_edges = self.extract_edges(image_threshold)
            if debug:
                plt.title(f"Section [{section}] Edges")
                plt.imshow(
                    image_edges,
                    cmap='binary', 
                    aspect='auto'
                )
                

            image_lines = self.extract_lines(image_edges)
            # Iterate over points
            for points in image_lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]

                if x1 == x2:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                
                if np.abs(slope)/self.image_ratio < 0.5:
                    continue
                y_middle = self.current_data_height//2
                y_upper = self.current_data_height
                y_lower = 0
                if y2 > y_middle:
                    px = self.x_intercept(x1, y1, x2, y2, y_upper)
                    py = y_upper
                else:
                    px = self.x_intercept(x1, y1, x2, y2, y_lower)
                    py = y_lower

                x_intercept = self.x_intercept(x1, y1, x2, y2, y_middle)
                # if x_intercept < 0 or x_intercept > self.current_data_width:
                #     continue

                if slope > 0:
                    line_dict[section]["negative"].append([px, py, x_intercept, y_middle])
                else:
                    line_dict[section]["positive"].append([px, py, x_intercept, y_middle])

                if debug:
                    if section == "upper":
                        color = 'blue'
                    else:
                        color = "red"

                    if slope > 0:
                        color = "dark" + color
                    else:
                        color = color

                    plt.plot([px, x_intercept], [py, y_middle], c=color)
            if debug:
                plt.show()

        upper_pos = sorted(line_dict['upper']['positive'], key = lambda x: x[2])
        upper_neg = sorted(line_dict['upper']['negative'], key = lambda x: x[2])
        lower_pos = sorted(line_dict['lower']['positive'], key = lambda x: x[2])
        lower_neg = sorted(line_dict['lower']['negative'], key = lambda x: x[2])

        assert len(upper_pos) == len(upper_neg), "Unbalanced lines detected in the upper half"
        assert len(lower_pos) == len(lower_neg), "Unbalanced lines detected in the lower half"

        diamond_shapes = []  
        for u_p, u_n, l_p, l_n in zip(upper_pos, upper_neg, lower_pos, lower_neg):
            # [2] is the x-intercept of the line

            left_x_int = [max(0, int((u_p[2] + l_n[2]) / 2)), self.current_data_height //2]
            right_x_int = [min(self.current_data_height, int((u_n[2] + l_p[2]) / 2)), self.current_data_height //2]
            upper_vertex = self.get_intersect(u_p, u_n)
            lower_vertex = self.get_intersect(l_n, l_p)

            diamond_shapes.append([left_x_int, upper_vertex, right_x_int, lower_vertex])

        for i in range(len(diamond_shapes) - 1):
            left_diamond = diamond_shapes[i]
            right_diamond = diamond_shapes[i+1]

            average_x = int((left_diamond[2][0] + right_diamond[0][0]) / 2)
            left_diamond[2][0] = average_x
            right_diamond[0][0] = average_x

        detected_coulomb_diamonds = []
        for number, diamond_shape in enumerate(diamond_shapes):
                    
            xs, ys = zip(*(diamond_shape+diamond_shape[:1]))
            gate_values = [self.gate_data[0] + x * self.gate_voltage_per_pixel for x in xs]
            ohmic_values = [y * self.ohmic_voltage_per_pixel - self.ohmic_data[-1] for y in ys]
            diamond_vertices_voltage = np.vstack((gate_values, ohmic_values)).T[:-1, :]

            detected_coulomb_diamonds.append(
                CoulombDiamond(
                    name=f"#{number}",
                    left_vertex=diamond_vertices_voltage[0],
                    top_vertex=diamond_vertices_voltage[3],
                    right_vertex=diamond_vertices_voltage[2],
                    bottom_vertex=diamond_vertices_voltage[1],
                    oxide_thickness=self.oxide_thickness,
                    epsR=self.epsR
                )
            )

        self.diamonds = detected_coulomb_diamonds
        return detected_coulomb_diamonds


    def estimate_temperatures(
        self, 
        diamonds: list[CoulombDiamond], 
        ohmic_value: float,
        temperature_guess: float = 1,
        debug = False) -> list[float]:
    
        fig, axes = plt.subplots()
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.375, 0.22, 0.3, 0.25]
        ax_inset = fig.add_axes([left, bottom, width, height])
        axes.imshow(
            self.current_data, 
            cmap='binary',
            aspect='auto',
            origin='lower',
            extent=[
                self.gate_data[0],
                self.gate_data[-1], 
                self.ohmic_data[0], 
                self.ohmic_data[-1]
            ],
        )
        
        axes.set_xlabel("Gate Voltage (V)")
        axes.set_ylabel("Ohmic Voltage (V)")
        axes.axhline(ohmic_value, c='k', linestyle="--")
        axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        temperatures = []
        for i in range(len(diamonds)-1):
            # Get estimated lever arm from neighbouring diamonds
            if debug:
                print(f"Processing Diamond {i}, name: {diamonds[i].name}")
            left_diamond = diamonds[i]
            right_diamond = diamonds[i+1]
            left_Vg, left_Vsd = left_diamond.top_vertex
            right_Vg, right_Vsd = right_diamond.top_vertex
            average_lever_arm = (left_diamond.lever_arm() + right_diamond.lever_arm()) / 2
            if debug:
                print(f"Left Vertex: {left_diamond.top_vertex}")
                print(f"Right Vertex: {right_diamond.top_vertex}")
                print(f"Average Lever Arm: {average_lever_arm}")
                print(f'Gate data: {self.gate_data}')
                print(f'Ohmic data: {self.ohmic_data}')

                for val in self.gate_data:

                    if val >= left_Vg and val <= right_Vg:
                        print(f"Val: {val} >= {left_Vg} and {val} <= {right_Vg}")
                    

            # Filter out the coulomb oscillation between two diamonds
            gate_mask = np.where((self.gate_data >= left_Vg) * (self.gate_data <= right_Vg))[0]
            ohmic_index = (np.abs(self.ohmic_data - ohmic_value)).argmin()
            P_data_filtered = self.gate_data[gate_mask]
            ohmic_value = self.ohmic_data[ohmic_index]
            oscillation = self.current_data[ohmic_index, gate_mask]
            if debug:
                print(f"Gate Mask: {gate_mask}")
                print(f"Gate Data: {P_data_filtered}")
                print(f"Ohmic Index: {ohmic_index}")
                print(f"Ohmic Value: {ohmic_value}")
                print(f"Oscillation: {oscillation}")

            # Fit data to Coulomb peak theoretical formula
            guess = [oscillation.min(), oscillation.max(), np.average(P_data_filtered), temperature_guess]
            (a,b,V0,Te), coeffs_cov = sp.optimize.curve_fit(
                lambda V, a, b, V0, Te: self.coulomb_peak(V, average_lever_arm, a, b, V0, Te), 
                P_data_filtered, 
                oscillation, 
                p0=guess,
                bounds=([-np.inf, -np.inf, -np.inf, -np.inf],[np.inf,np.inf,np.inf,np.inf])
            )

            # Plot results
            Vs = np.linspace(P_data_filtered.min(), P_data_filtered.max(), 100)
            ax_inset.plot(P_data_filtered, oscillation, 'k.')
            ax_inset.plot(Vs, self.coulomb_peak(Vs, average_lever_arm, a, b, V0, Te),'k-')
            ax_inset.set_xlabel(r"Gate Voltage (V)")
            ax_inset.set_ylabel(r"Current (A)")

            temperatures.append(Te)
        temperatures = np.array(temperatures)
        T_avg = np.average(np.abs(temperatures))
        T_stddev = np.std(temperatures)
    
        axes.set_title(
            f'$T_e$ = ${round(T_avg,3)}$ K $\pm\ {round(T_stddev/np.sqrt(len(temperatures)), 3)}$ K'
        )
                
        plt.show()
        return temperatures

    def coulomb_peak(self, V, alpha, a, b, V0, Te):
        kB = 8.6173303e-5 # eV / K
        return a + b * (np.cosh(alpha * (V0 - V) / (2 * kB * Te)))**(-2)

    def get_statistics(self, diamonds: Optional[list[CoulombDiamond]] = None) -> dict:
        if diamonds is None:
            diamonds = self.diamonds

        methods = [
            'lever_arm',
            'addition_voltage',
            'charging_voltage',
            'total_capacitance',
            'gate_capacitance',
            'source_capacitance',
            'drain_capacitance',
            'dot_radius',
        ]
        methods_print = {
            'lever_arm': lambda mu, std: f"Average Lever Arm (\u03B1) : {mu:.5f} eV/V \u00b1 {std:.5f} eV/V",
            'addition_voltage' : lambda mu, std: f"Average Addition Voltage: {1e3 * mu:.5f} mV \u00b1 {1e3 *std:.5f} mV",
            'charging_voltage': lambda mu, std: f"Average Charging Voltage: {1e3 *mu:.5f} mV \u00b1 {1e3 *std:.5f} mV",
            'total_capacitance': lambda mu, std: f"Average Total Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'gate_capacitance': lambda mu, std: f"Average Gate Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'source_capacitance': lambda mu, std: f"Average Source Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'drain_capacitance': lambda mu, std: f"Average Drain Capacitance: {1e18 * mu:.5f} aF \u00b1 {1e18 * std:.5f} aF",
            'dot_radius': lambda mu, std: f"Average Dot Radius: {1e9 * mu:.5f} nm \u00b1 {1e9 * std:.5f} nm",
        }


        results = {}

        for method in methods:
            results[method] = sp.stats.norm.fit([getattr(diamond, method)() for diamond in diamonds])
        
        for method, (mu, std) in results.items():
            print(
                methods_print[method](mu, std/len(diamonds))
            )

        return results
    
    def extract_edges(self, image: ndarray, threshold1: int = 0, threshold2: int = 0, apertureSize: int = 3) -> ndarray:

        # Perform Canny edge detection
        edges = cv2.Canny(
            image, 
            threshold1, 
            threshold2, 
            apertureSize=apertureSize
        )

        return edges

    def extract_lines(self, 
                      edges: ndarray,
                      threshold: int = 10,
                      maxLineGap: int = 250,
                      minLineLength: int = None,
                      distanceThreshold: int = None,
                      angleThreshold: float = None
                      ) -> list:

        if minLineLength is None:
            minLineLength = self.current_data_height // 10
        if distanceThreshold is None:
            distanceThreshold = self.current_data_width//10
        if angleThreshold is None:
            angleThreshold = 0.25
        lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=threshold, # Min number of votes for valid line
            minLineLength=minLineLength, # Min allowed length of line
            maxLineGap=maxLineGap # Max allowed gap between line for joining them
            )
        # Filter out duplicate lines
        lines = self.filter_duplicate_lines(lines, distance_threshold=distanceThreshold, angle_threshold=angleThreshold)

        return lines
    
    def process_image(self, data,
                    rescaling_factor: int = 2, 
                    morph_kernel_size: Optional[tuple] = (5, 5), 
                    morph_iterations: Optional[int] = 2, 
                    blur_kernel_size: Optional[tuple] = (7, 7), 
                    blur_sigma: Optional[float] = 1, 
                    threshold_value: Optional[int] = 200,
                    second_blur = False,
                    skip_processing = False) -> None:
        """
        Increase the resolution of self.current_data by interpolating.

        Perfforms different enhancement operations on the data such as morphological operations, Gaussian blurring, and thresholding.
        
        Parameters:
        rescalingfactor (int): The factor by which to increase the resolution.
        morph_kernel_size (tuple): The size of the kernel for morphological operations.
        morph_iterations (int): The number of iterations for morphological operations.
        blur_kernel_size (tuple): The size of the kernel for Gaussian blurring.
        blur_sigma (float): The sigma value for Gaussian blurring.
        threshold_value (int): The threshold value for binary thresholding.
        second_blur (bool): Whether to perform a second Gaussian blur operation.
        skip_processing (bool): Whether to skip processing and return the resized data.
        """
        if skip_processing:
            return data

        new_height = self.current_data_height * rescaling_factor
        new_width = self.current_data_width * rescaling_factor

        
        # Interpolate the current data to the new resolution
        resized = cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Perform closing if morph_kernel_size and morph_iterations are provided
        if morph_kernel_size is not None and morph_iterations is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
            resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        # Perform Gaussian blurring if blur_kernel_size and blur_sigma are provided
        if blur_kernel_size is not None and blur_sigma is not None and second_blur:
            resized = cv2.GaussianBlur(resized, blur_kernel_size, blur_sigma)

        # Apply thresholding if threshold_value is provided
        if threshold_value is not None:
            _, resized = cv2.threshold(resized, threshold_value, 255, cv2.THRESH_BINARY)

        # Update the dimensions and voltage per pixel values
        self.current_data_height, self.current_data_width = self.current_data.shape
        self.gate_voltage_per_pixel /= rescaling_factor
        self.ohmic_voltage_per_pixel /= rescaling_factor
        return resized
    
    def extract_diamonds_direct(self, 
                                debug: bool = False,
                                center_offset: int = 0,
                                naive_extraction: bool = False,
                                resolution_factor: int = 2,
                                min_area: int = 500,
                                morph_iterations: int = 2,
                                morph_kernel = (5,5),
                                second_threshold = 200,
                                second_blur = False,
                                skip_advanced_processing = False) -> None:  
        
    
        # Run filter_raw_data on the full dataset
        filtered_data = self.filter_raw_data(
            self.current_data,
            binary_threshold=self.binary_threshold,
            binary_threshold_linear=self.binary_threshold_linear,
            blur_sigma=self.blur_sigma,
            blur_kernel=self.blur_kernel
        )

        # Calculate the center with the offset
        center = self.current_data_height // 2 + center_offset

        # if debug:
        #     plt.title("Filtered Data")
        #     plt.imshow(filtered_data, cmap='binary', aspect='auto', origin='lower')
        #     plt.hlines(center, 0, self.current_data_width, colors='red')
        #     plt.show()

        
        processed_data = self.process_image(data=filtered_data,
                                           rescaling_factor=resolution_factor, 
                                           morph_kernel_size=morph_kernel, 
                                           morph_iterations=morph_iterations, 
                                           blur_kernel_size=self.blur_kernel, 
                                           blur_sigma=self.blur_sigma, 
                                           threshold_value=second_threshold,
                                           second_blur=second_blur,
                                           skip_processing=skip_advanced_processing)


        # Add a padding to the binary image
        border_size = 10  # Adjust as needed
        padded_data = cv2.copyMakeBorder(processed_data, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)

        # # Calculate the center with the offset
        center_processed = self.current_data_height // 2 * resolution_factor + (center_offset * resolution_factor) + border_size

        if debug:
            plt.title("Pre-processed Data")
            plt.imshow(padded_data, cmap='binary', aspect='auto',origin='lower')
            plt.hlines(center_processed, 0, (self.current_data_width + border_size)* resolution_factor , colors='red')
            plt.show()


        contours, hierarchy = cv2.findContours(padded_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # #Filter out contours that are too small by area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # closed_contours = filtered_contours





        # Sort the closed contours from left to right based on their minimum x-coordinate
        sorted_contours = sorted(filtered_contours, key=lambda cnt: cnt[:, :, 0].min())

        


        # Adjust contour points after removing padding
        adjusted_contours = []
        for contour in sorted_contours:
            adjusted_contour = contour - border_size
            adjusted_contours.append(adjusted_contour)

        if debug:
            plt.title("Filtered Contours")
            for i, contour in enumerate(adjusted_contours):
                plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
                plt.text(contour[0][0][0], contour[0][0][1], f"#{i}")
            plt.show()




        # print(extraction_data.shape, center_processed)

        linecut = padded_data[center_processed,:]




        # Find the left and right vertices using the linecut
        left_vertices = []
        right_vertices = []
        in_pulse = False
        for i, value in enumerate(linecut):
            if value > 0 and not in_pulse:
                in_pulse = True
                left_vertices.append((i- border_size, center_processed - border_size))
            elif value == 0 and in_pulse:
                in_pulse = False
                right_vertices.append((i- border_size, center_processed - border_size))                      
            
            if len(right_vertices) >= len(adjusted_contours):
                break      
    
                # # Remove the border to restore the original image size
        extraction_data = padded_data[border_size:-border_size, border_size:-border_size]
        center_processed = center_processed - border_size

        

     
        top_vertices = []
        bottom_vertices = []
        for contour in adjusted_contours:
            top_tip = tuple(contour[contour[:, :, 1].argmin()][-1])
            bottom_tip = tuple(contour[contour[:, :, 1].argmax()][-1])            
            top_vertices.append(top_tip)
            bottom_vertices.append(bottom_tip)

        if naive_extraction:
            # Find width and height of each closed contour
            left_vertices_var = []
            right_vertices_var = []
            top_vertices_var = []
            bottom_vertices_var = []
            
            for contour in adjusted_contours:
                x, y, w, h = cv2.boundingRect(contour)
                center = y + h // 2
                left_vertex = (x, center)
                right_vertex = (x + w, center)
                top_vertex = (x + w // 2, y)
                bottom_vertex = (x + w // 2, y + h)
                left_vertices_var.append(left_vertex)
                right_vertices_var.append(right_vertex)
                top_vertices_var.append(top_vertex)
                bottom_vertices_var.append(bottom_vertex)       

            left_vertices = left_vertices_var
            right_vertices = right_vertices_var
            top_vertices = top_vertices_var
            bottom_vertices = bottom_vertices_var       

        if debug:
            plt.title("Vertices")
            plt.imshow(extraction_data, cmap='binary', aspect='auto', origin='lower')
            for i in range(len(left_vertices)):
                plt.hlines(center_processed, 0, (self.current_data_width) * resolution_factor, colors='red')
                plt.plot(left_vertices[i][0], left_vertices[i][1], 'go')  # Green for left vertices
                plt.plot(top_vertices[i][0], top_vertices[i][1], 'ro')  # Red for top vertices
                plt.plot(right_vertices[i][0], right_vertices[i][1], 'bo')  # Blue for right vertices
                plt.plot(bottom_vertices[i][0], bottom_vertices[i][1], 'yo')  # Yellow for bottom vertices

        if any([len(left_vertices) != len(right_vertices), len(left_vertices) != len(top_vertices), len(left_vertices) != len(bottom_vertices)]):
            raise ValueError(f"The number of vertices detected does not match. \n Left: {len(left_vertices)} \n Right: {len(right_vertices)} \n Top: {len(top_vertices)} \n Bottom: {len(bottom_vertices)}")

        detected_coulomb_diamonds = []
        for number in range(len(left_vertices)):
            left_vertex = left_vertices[number]
            right_vertex = right_vertices[number]
            top_vertex = bottom_vertices[number]  # Inverted due to the image being inverted
            bottom_vertex = top_vertices[number]
            if debug:
                plt.plot(left_vertex[0], left_vertex[1], 'go')  # Green for left vertex
                plt.plot(top_vertex[0], top_vertex[1], 'ro')  # Red for top vertex
                plt.plot(right_vertex[0], right_vertex[1], 'bo')  # Blue for right vertex
                plt.plot(bottom_vertex[0], bottom_vertex[1], 'yo')  # Yellow for bottom vertex

            gate_values = [self.gate_data[0] + v[0] * self.gate_voltage_per_pixel for v in [left_vertex, top_vertex, right_vertex, bottom_vertex]]
            ohmic_values = [v[1] * self.ohmic_voltage_per_pixel - self.ohmic_data[-1] for v in [left_vertex, top_vertex, right_vertex, bottom_vertex]]
            diamond_vertices_voltage = np.vstack((gate_values, ohmic_values)).T

            detected_coulomb_diamonds.append(
            CoulombDiamond(
                name=f"{number}",
                left_vertex=diamond_vertices_voltage[0],
                top_vertex=diamond_vertices_voltage[1],
                right_vertex=diamond_vertices_voltage[2],
                bottom_vertex=diamond_vertices_voltage[3],
                oxide_thickness=self.oxide_thickness,
                epsR=self.epsR
            )
            )
        plt.show()
        self.diamonds = detected_coulomb_diamonds
        return detected_coulomb_diamonds
    

## GOOD VERSION
# def extract_diamonds_direct(self, 
#                                 debug: bool = False,
#                                 center_offset: int = 0,
#                                 edge_threshold_low: int = 0,
#                                 edge_threshold_high: int = 0,
#                                 naive_extraction: bool = False,
#                                 resolution_factor: int = 2,
#                                 min_area: int = 100,
#                                 contour_epsilon_factor: float = 0.02,
#                                 morph_iterations: int = 2) -> None:  
        
    
#         # Run filter_raw_data on the full dataset
#         filtered_data = self.filter_raw_data(
#             self.current_data,
#             binary_threshold=self.binary_threshold,
#             binary_threshold_linear=self.binary_threshold_linear,
#             blur_sigma=self.blur_sigma,
#             blur_kernel=self.blur_kernel
#         )

#         # Calculate the center with the offset
#         center = self.current_data_height // 2 + center_offset

#         if debug:
#             plt.title("Filtered Data")
#             plt.imshow(filtered_data, cmap='binary', aspect='auto')
#             plt.hlines(center, 0, self.current_data_width, colors='red')
#             plt.show()

        
#         processed_data = self.process_image(data=filtered_data,
#                                            rescaling_factor=resolution_factor, 
#                                            morph_kernel_size=(5, 5), 
#                                            morph_iterations=morph_iterations, 
#                                            blur_kernel_size=(7, 7), 
#                                            blur_sigma=1, 
#                                            threshold_value=200)

#         # if debug:
#         #     plt.title("Filtered Data Connected")
#         #     plt.imshow(filtered_data, cmap='binary', aspect='auto')
#         #     plt.hlines(center, 0, self.current_data_width, colors='red')
#         #     plt.show()

#         # Add a padding to the binary image
#         border_size = 10  # Adjust as needed
#         padded_data = cv2.copyMakeBorder(processed_data, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)

#         # # Calculate the center with the offset
#         center_processed = self.current_data_height // 2 * resolution_factor + (center_offset * resolution_factor) + border_size

#         if debug:
#             plt.title("Enhanced Filtered Data")
#             plt.imshow(padded_data, cmap='binary', aspect='auto')
#             plt.hlines(center_processed, 0, self.current_data_width * resolution_factor, colors='red')
#             plt.show()

#         # edges = self.extract_edges(filtered_data,
#         #                            threshold1=edge_threshold_low,
#         #                            threshold2=edge_threshold_high,
#         #                            apertureSize=3)        
        
#         # if debug:
#         #     plt.title("Edges")
#         #     plt.imshow(filtered_data, cmap='binary', aspect='auto')
#         #     plt.hlines(center, 0, self.current_data_width, colors='red')
#         #     plt.show()
        
#         contours, hierarchy = cv2.findContours(padded_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # if debug:
#         #     plt.title("Contours")
#         #     for i,contour in enumerate(contours):
#         #         plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
#         #     plt.show()

#         # contours_approx = []
#         # for contour in contours:
#         #     # Approximate the contour to simplify its shape
#         #     epsilon = contour_epsilon_factor * cv2.arcLength(contour, True)
#         #     contours_approx.append(cv2.approxPolyDP(contour, epsilon, True))

#         # if debug:
#         #     plt.title("Approximated Contours")
#         #     for i, contour in enumerate(contours_approx):
#         #         plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
#         #     plt.show()


#         # #Filter out contours that do not form a closed shape
#         # closed_contours = [cnt for i, cnt in enumerate(contours_approx) if hierarchy[0][i][2] != -1 and cv2.contourArea(cnt) > 0]

#         # if debug:
#         #     plt.title("Closed Contours")
#         #     for contour in closed_contours:
#         #         plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
#         #     plt.show()

#         # closed_contours = contours

#         # for contour in closed_contours:
#         #     print(cv2.contourArea(contour))
#         # #Filter out contours that are too small by area
#         filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

#         # closed_contours = filtered_contours





#         # if debug:
#         #     plt.title("Filtered Contours")
#         #     for i, contour in enumerate(closed_contours):
#         #         plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
#         #         plt.text(contour[0][0][0], contour[0][0][1], f"Contour #{i}")
#         #     plt.show()
        

#         # # Create a blank image to draw the closed contours
#         # closed_contours_image = np.zeros_like(edges)

#         # # Draw the closed contours on the blank image
#         # cv2.drawContours(closed_contours_image, contours, -1, (255), thickness=cv2.FILLED)


#         # edges_closed = closed_contours_image

#         # # Remove the border to restore the original image size
#         # edges_closed = edges_closed[border_size:-border_size, border_size:-border_size]

#         # if debug:
#         #     plt.title("Edges")
#         #     plt.imshow(edges_closed, cmap='binary', aspect='auto')
#         #     plt.show()

#         # Sort the closed contours from left to right based on their minimum x-coordinate
#         sorted_contours = sorted(filtered_contours, key=lambda cnt: cnt[:, :, 0].min())

#         # # Remove the border to restore the original image size
#         extraction_data = padded_data[border_size:-border_size, border_size:-border_size]


#         # Adjust contour points after removing padding
#         adjusted_contours = []
#         for contour in sorted_contours:
#             adjusted_contour = contour - border_size
#             adjusted_contours.append(adjusted_contour)

#         if debug:
#             plt.title("Adjusted Contours")
#             for i, contour in enumerate(adjusted_contours):
#                 plt.plot(contour[:, :, 0], contour[:, :, 1], 'r')
#                 plt.text(contour[0][0][0], contour[0][0][1], f"Contour #{i}")
#             plt.show()

#         # if debug:
#         #     plt.title("Extraction Data")
#         #     plt.imshow(extraction_data, cmap='binary', aspect='auto')
#         #     plt.hlines(center_processed, 0, self.current_data_width * resolution_factor, colors='red')
#         #     plt.show()

#         center_processed = center_processed - border_size

#         # print(extraction_data.shape, center_processed)

#         linecut = extraction_data[center_processed,:]

#         # if debug:
#         #     plt.title("Extraction Data")
#         #     plt.imshow(extraction_data, cmap='binary', aspect='auto')
#         #     plt.hlines(center_processed, 0, self.current_data_width * resolution_factor, colors='red')
#         #     plt.show()

#         #     plt.title("Linecut")
#         #     plt.plot(linecut)
#         #     plt.show()



#         # Find the left and right vertices using the linecut
#         left_vertices = []
#         right_vertices = []
#         in_pulse = False
#         for i, value in enumerate(linecut):
#             if value > 0 and not in_pulse:
#                 in_pulse = True
#                 left_vertices.append((i, center_processed))
#             elif value == 0 and in_pulse:
#                 in_pulse = False
#                 right_vertices.append((i, center_processed))
            



#         top_vertices = []
#         bottom_vertices = []
#         for contour in adjusted_contours:
#             top_tip = tuple(contour[contour[:, :, 1].argmin()][0])
#             bottom_tip = tuple(contour[contour[:, :, 1].argmax()][0])            
#             top_vertices.append(top_tip)
#             bottom_vertices.append(bottom_tip)

#         if naive_extraction:
#             # Find width and height of each closed contour
#             left_vertices_var = []
#             right_vertices_var = []
#             top_vertices_var = []
#             bottom_vertices_var = []
            
#             for contour in adjusted_contours:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 center = y + h // 2
#                 left_vertex = (x, center)
#                 right_vertex = (x + w, center)
#                 top_vertex = (x + w // 2, y)
#                 bottom_vertex = (x + w // 2, y + h)
#                 left_vertices_var.append(left_vertex)
#                 right_vertices_var.append(right_vertex)
#                 top_vertices_var.append(top_vertex)
#                 bottom_vertices_var.append(bottom_vertex)       

#             left_vertices = left_vertices_var
#             right_vertices = right_vertices_var
#             top_vertices = top_vertices_var
#             bottom_vertices = bottom_vertices_var       

#         if debug:
#             plt.title("Vertices")
#             plt.imshow(extraction_data, cmap='binary', aspect='auto')
#             for i in range(len(left_vertices)):
#                 plt.hlines(center_processed, 0, self.current_data_width * resolution_factor, colors='red')
#                 plt.plot(left_vertices[i][0], left_vertices[i][1], 'go')  # Green for left vertices
#                 plt.plot(top_vertices[i][0], top_vertices[i][1], 'ro')  # Red for top vertices
#                 plt.plot(right_vertices[i][0], right_vertices[i][1], 'bo')  # Blue for right vertices
#                 plt.plot(bottom_vertices[i][0], bottom_vertices[i][1], 'yo')  # Yellow for bottom vertices

#         if any([len(left_vertices) != len(right_vertices), len(left_vertices) != len(top_vertices), len(left_vertices) != len(bottom_vertices)]):
#             raise ValueError(f"The number of vertices detected does not match. \n Left: {len(left_vertices)} \n Right: {len(right_vertices)} \n Top: {len(top_vertices)} \n Bottom: {len(bottom_vertices)}")

#         detected_coulomb_diamonds = []
#         for number in range(len(left_vertices)):
#             left_vertex = left_vertices[number]
#             right_vertex = right_vertices[number]
#             top_vertex = bottom_vertices[number]  # Inverted due to the image being inverted
#             bottom_vertex = top_vertices[number]
#             if debug:
#                 plt.plot(left_vertex[0], left_vertex[1], 'go')  # Green for left vertex
#                 plt.plot(top_vertex[0], top_vertex[1], 'ro')  # Red for top vertex
#                 plt.plot(right_vertex[0], right_vertex[1], 'bo')  # Blue for right vertex
#                 plt.plot(bottom_vertex[0], bottom_vertex[1], 'yo')  # Yellow for bottom vertex

#             gate_values = [self.gate_data[0] + v[0] * self.gate_voltage_per_pixel for v in [left_vertex, top_vertex, right_vertex, bottom_vertex]]
#             ohmic_values = [v[1] * self.ohmic_voltage_per_pixel - self.ohmic_data[-1] for v in [left_vertex, top_vertex, right_vertex, bottom_vertex]]
#             diamond_vertices_voltage = np.vstack((gate_values, ohmic_values)).T

#             detected_coulomb_diamonds.append(
#             CoulombDiamond(
#                 name=f"{number}",
#                 left_vertex=diamond_vertices_voltage[0],
#                 top_vertex=diamond_vertices_voltage[1],
#                 right_vertex=diamond_vertices_voltage[2],
#                 bottom_vertex=diamond_vertices_voltage[3],
#                 oxide_thickness=self.oxide_thickness,
#                 epsR=self.epsR
#             )
#             )
#         plt.show()
#         self.diamonds = detected_coulomb_diamonds
#         return detected_coulomb_diamonds

    def filter_duplicate_lines(self, lines, distance_threshold=None, angle_threshold=None):
        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            keep_line = True
            if y2 == y1:
                continue
            for other_line in filtered_lines:
                ox1, oy1, ox2, oy2 = other_line[0]
                # Calculate the slopes of the two lines
                slope1 = np.arctan2((y2 - y1), (x2 - x1))
                slope2 = np.arctan2((oy2 - oy1), (ox2 - ox1))
                if np.sign(slope1) != np.sign(slope2):
                    # slopes are opposite, definitely not similar
                    continue
                if self.are_lines_similar(x1, y1, x2, y2, ox1, oy1, ox2, oy2, distance_threshold, angle_threshold):
                    keep_line = False
                    break
            if keep_line:
                filtered_lines.append(line)

        return filtered_lines

    def get_intersect(self, l1, l2):
        l1_x1, l1_y1, l1_x2, l1_y2 = l1
        l2_x1, l2_y1, l2_x2, l2_y2 = l2
        s = np.vstack([[l1_x1, l1_y1], [l1_x2, l1_y2], [l2_x1, l2_y1], [l2_x2, l2_y2]])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return [float('inf'), float('inf')]
        else:
            return [int(x/z), int(y/z)]

    def are_lines_similar(self, x1, y1, x2, y2, ox1, oy1, ox2, oy2, distance_threshold, angle_threshold):

        y_middle = self.current_data_height//2
        y_upper = self.current_data_height
        y_lower = 0
        
        if y2 > y_middle:
            px = self.x_intercept(x1, y1, x2, y2, y_upper)
        else:
            px = self.x_intercept(x1, y1, x2, y2, y_lower)

        if oy2 > y_middle:
            opx = self.x_intercept(ox1, oy1, ox2, oy2, y_upper)
        else:
            opx = self.x_intercept(ox1, oy1, ox2, oy2, y_lower)

        distance_difference = np.abs(px - opx)

        # Calculate the slopes of the two lines
        slope1 = np.arctan2((y2 - y1), (x2 - x1))
        slope2 = np.arctan2((oy2 - oy1), (ox2 - ox1))
        angle_difference = np.abs(slope1 - slope2)%(2*np.pi)

        # Check if both distance and angle difference are below the thresholds
        return distance_difference < distance_threshold and angle_difference < angle_threshold

    def x_intercept(self, x1, y1, x2, y2, y_i):
        # Check if the line is vertical to avoid division by zero
        if y2 == y1:
            raise ValueError("The line defined by the points is horizontal and does not intercept the x-axis.")
        if x1 == x2:
            return int(x1)
        # Calculate the slope
        m = (y2 - y1) / (x2 - x1)
        # Calculate the x-intercept
        x_i = (y_i - y1) / m + x1
        
        return int(x_i)
    
    def plot_diamonds(self):

        fig, ax = plt.subplots()
        ax.imshow(
            self.current_data, 
            cmap='binary',
            aspect='auto',
            origin='lower',
            extent=[
                self.gate_data[0],
                self.gate_data[-1], 
                self.ohmic_data[0], 
                self.ohmic_data[-1]
            ],
        )
        ax.set_title("Coulomb Diamonds")
        ax.set_xlabel("Gate Voltage (V)")
        ax.set_ylabel("Ohmic Voltage (V)")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        for diamond in self.diamonds:
            diamond.print_summary()
            diamond.plot(ax)

        plt.tight_layout()
        plt.show()
