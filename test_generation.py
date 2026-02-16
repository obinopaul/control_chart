#!/usr/bin/env python3
import numpy as np

def test_parameter_ranges():
    """Test the new parameter ranges according to advisor's specifications"""
    
    # Generate exactly num_values unique t values, ensuring start and stop values are included
    def generate_exact_unique_t_values(start, stop, num_values, precision=2):
        if num_values == 2:
            return np.array([start, stop])

        # Generate values between start and stop
        raw_values = np.linspace(start, stop, num_values)
        
        # Round the values to the required precision
        rounded_values = np.round(raw_values, precision)
        
        # Ensure start and stop values are exactly included
        rounded_values[0] = np.round(start, precision)
        rounded_values[-1] = np.round(stop, precision)
        
        # If there are duplicates due to rounding, increase the precision
        if len(np.unique(rounded_values)) < num_values:
            return generate_exact_unique_t_values(start, stop, num_values, precision + 1)
        
        return rounded_values

    # Define the new parameter ranges
    t_ranges = {
        1: generate_exact_unique_t_values(0.005, 0.605, 20),  # Uptrend - slope k: [0.005r, 0.605r]
        2: generate_exact_unique_t_values(0.005, 0.605, 20),  # Downtrend - slope k: [0.005r, 0.605r] 
        3: generate_exact_unique_t_values(0.005, 1.805, 20),  # Upshift - shift x: [0.005r, 1.805r]
        4: generate_exact_unique_t_values(0.005, 1.805, 20),  # Downshift - shift x: [0.005r, 1.805r]
        5: generate_exact_unique_t_values(0.005, 1.805, 20),  # Systematic - systematic k: [0.005r, 1.805r]
        6: generate_exact_unique_t_values(0.005, 1.805, 20),  # Cyclic - cyclic a: [0.005r, 1.805r]
        7: generate_exact_unique_t_values(0.005, 0.8, 20)     # Stratification - std dev e_t: [0.005r, 0.8r]
    }
    
    print("New Parameter Ranges (with r=1):")
    pattern_names = {
        1: 'Uptrend (slope k)',
        2: 'Downtrend (slope k)', 
        3: 'Upshift (shift x)',
        4: 'Downshift (shift x)',
        5: 'Systematic (systematic k)',
        6: 'Cyclic (cyclic a)',
        7: 'Stratification (std dev e_t)'
    }
    
    for abtype, values in t_ranges.items():
        name = pattern_names[abtype]
        print(f"  Pattern {abtype} - {name}:")
        print(f"    Range: [{values.min():.3f}, {values.max():.3f}]")
        print(f"    Values: {len(values)} points")
        print(f"    Sample values: {values[:3]} ... {values[-3:]}")
        print()

def test_data_generation_equations():
    """Test the data generation equations"""
    print("Testing Data Generation Equations:")
    print("=================================")
    
    # Test parameters
    w = 10  # window length
    t = 0.5  # test parameter value
    
    # Time series index
    s = np.arange(1, w + 1)
    
    print(f"Window length (w): {w}")
    print(f"Time index (s): {s}")
    print(f"Test parameter (t): {t}")
    print()
    
    # Test each pattern
    patterns = {
        1: "Uptrend",
        2: "Downtrend", 
        3: "Upshift",
        4: "Downshift",
        5: "Systematic",
        6: "Cyclic",
        7: "Stratification"
    }
    
    for abtype, name in patterns.items():
        print(f"Pattern {abtype} - {name}:")
        
        # Generate normal baseline
        normal_data = np.random.randn(w) * 1 + 0  # mean=0, std=1
        
        if abtype == 1:  # Uptrend
            abnormal_data = 0 + np.random.randn(w) * 1 + t * s
            print(f"  Equation: 0 + N(0,1) + {t} * s")
            
        elif abtype == 2:  # Downtrend
            abnormal_data = 0 + np.random.randn(w) * 1 - t * s
            print(f"  Equation: 0 + N(0,1) - {t} * s")
            
        elif abtype == 3:  # Upshift
            shift_point = w // 2
            abnormal_data = np.zeros(w)
            abnormal_data[:shift_point] = np.random.randn(shift_point) * 1 + 0
            abnormal_data[shift_point:] = np.random.randn(w - shift_point) * 1 + 0 + t
            print(f"  Equation: 0 + N(0,1) before point {shift_point}, then 0 + N(0,1) + {t}")
            
        elif abtype == 4:  # Downshift
            shift_point = w // 2
            abnormal_data = np.zeros(w)
            abnormal_data[:shift_point] = np.random.randn(shift_point) * 1 + 0
            abnormal_data[shift_point:] = np.random.randn(w - shift_point) * 1 + 0 - t
            print(f"  Equation: 0 + N(0,1) before point {shift_point}, then 0 + N(0,1) - {t}")
            
        elif abtype == 5:  # Systematic
            abnormal_data = 0 + np.random.randn(w) * 1 + t * (-1) ** s
            print(f"  Equation: 0 + N(0,1) + {t} * (-1)^s")
            
        elif abtype == 6:  # Cyclic
            period = 8
            abnormal_data = 0 + np.random.randn(w) * 1 + t * np.sin(2 * np.pi * s / period)
            print(f"  Equation: 0 + N(0,1) + {t} * sin(2π*s/{period})")
            
        elif abtype == 7:  # Stratification
            abnormal_data = 0 + np.random.randn(w) * t
            print(f"  Equation: 0 + N(0,{t})")
        
        print(f"  Sample result: {abnormal_data[:5]}")
        print()

if __name__ == "__main__":
    test_parameter_ranges()
    print("\n" + "="*50 + "\n")
    test_data_generation_equations()