""" This code creates all comparison metrics cubes in one go, using a registry of metrics. 

It is assumed that temperature, relative humidity, vapor pressure, dew point temperature
and wet-bulb temperature cubes have already been created and masked to the region of interest.
It is further assumed that the input cubes have been time zone corrected.

These inputs must be saved in the following format:
- input_dir/tas_masked.nc
- input_dir/rh_masked.nc
- input_dir/es_masked.nc
- input_dir/td_masked.nc
- input_dir/wbt_masked.nc

If required, the below are the conversions used to generate these cubes from tas and rh:
def dew_point_from_T_RH(temp_C, RH):
    gamma = np.log(RH / 100) + (17.27 * temp_C) / (237.3 + temp_C)
    return (237.3 * gamma) / (17.27 - gamma)

def vapor_pressure_from_T_RH(temp_C, RH):
    td = dew_point_from_T_RH(temp_C, RH)
    return 0.6108 * np.exp((17.27 * td) / (237.3 + td))

def wetbulb_temp_from_T_RH(temp_C, RH):
    Tw = temp_C * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) \
         + np.arctan(temp_C + RH) \
         - np.arctan(RH - 1.676331) \
         + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) \
         - 4.686035
    return Tw

"""

import numpy as np
from pathlib import Path
import iris
import math
from dataclasses import dataclass
from typing import Callable, List, Dict
import gc

def wetbulb_temp_from_tas_rh(tas, rh):
    if tas.units != 'Celsius':
        tas.convert_units('Celsius')
    Tw = tas * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) \
         + np.arctan(tas + rh) \
         - np.arctan(rh - 1.676331) \
         + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) \
         - 4.686035
    return Tw

def algorithm_1(tas, td):
    data = -2.653 + (0.994 * tas) + (0.0153 * (td ** 2))
    return data

def algorithm_2(tas, rh):
    data = -42.379 + (2.04901523 * tas) + (10.14333127 * rh) - (0.22475541 * tas * rh) - (6.83783e-3 * tas ** 2) - (5.471717e-2 * rh ** 2) + (1.22874e-3 * tas ** 2 * rh) + (8.5282e-4 * tas * (rh ** 2)) - ((1.99e-6 * (tas ** 2) * (rh ** 2)))
    return data

def algorithm_3(tas, rh):
    A = -10.3 + (1.1 * tas) + (0.047 * rh)
    B = (-42.379 + (2.04901523 * tas) + (10.14333127 * rh) - (0.22475541 * tas * rh) - (6.83783e-3 * (tas ** 2)) - (5.481717e-2 * (rh ** 2)) + (1.22874e-3 * (tas ** 2) * rh) + (8.5282e-4 * tas * (rh ** 2)) - (1.99e-6 * (tas ** 2) * (rh ** 2)))
    data = np.where(tas <= 40, tas,
                    np.where(A < 79, A,
                        np.where((rh < 13) & (80 <= tas) & (tas <= 112),
                                B - ((13 - rh) / 4) * ((17 - (tas - 95)) / 17) ** 0.5,
                                np.where((rh > 85) & (80 <= tas) & (tas <= 87),
                                        B + 0.02 * (rh - 85) * (87 - tas),
                                        B
                                )
                            )
                        )
                    )
    return data

def algorithm_4(tas, td):
    data = np.where(tas < 25, tas, -2.719 + (0.994 * tas) + (0.016 * (td ** 2)))
    return data

def algorithm_5(tas, rh):
    data = np.where(tas <= 20, tas, -8.784695 + (1.161139411 * tas) + (2.338549 * rh) - (0.14611605 * tas * rh) - (1.2308094e-2 * (tas ** 2)) - (1.6424828e-2 * (rh ** 2)) + (2.211732e-3 * (tas ** 2) * rh) + (7.2546e-4 * tas * (rh ** 2)) - (3.582e-6 * (tas ** 2) * (rh ** 2)))
    return data

def algorithm_6(tas, es):
    data = -1.3 + (0.92 * tas) + (2.2 * es)
    return data

def algorithm_7(tas, td):
    data = tas - 1.0799 * (math.e ** (0.03755 * tas)) * (1 - (math.e ** (0.0801 * (td - 14))))
    return data

def algorithm_8(tas, rh):
    data = np.where(tas < 75, tas, 16.923 + (0.185212 * tas) + (5.37941 * rh) - (0.100254 * tas * rh) + (9.4169e-3 * (tas ** 2)) + (7.28898e-3 * (rh ** 2)) + (3.45372e-4 * (tas ** 2) * rh) - (8.14971e-4 * tas * (rh ** 2)) + (1.02102e-5 * (tas ** 2) * (rh ** 2)) - (3.8646e-5 * (tas ** 3)) + (2.91583e-5 * (rh ** 3)) + (1.42721e-6 * (tas ** 3) * rh) + (1.97483e-7 * tas * (rh ** 3)) - (2.18429e-8 * (tas ** 3) * (rh ** 2)) + (8.43296e-10 * (tas ** 2) * (rh ** 3)) - (4.81975e-11 * (tas ** 3) * (rh ** 3)) + 0.5)
    return data

def algorithm_9(tas, rh, es):
    tr = (0.8841*tas) + 0.19
    p = (0.0196 * tas) + 0.9031
    data = tr + ((tas - tr) * ((((rh / 100) * es) / 1.6) ** p))
    return data

def algorithm_10(tas, wbt):
    data = (0.4 * wbt) + (0.4 * tas) + 15
    return data

def algorithm_11(tas, td):
    data = (0.55 * tas) + (0.2 * td) + 17.5
    return data

def algorithm_12(tas, wbt):
    data = (0.5 * wbt) + (0.5 * tas)
    return data

def algorithm_13(tas, rh):
    data = tas - ((0.55 * (1 - (0.01 * rh))) * (tas - 58))
    return data

def algorithm_14(tas, rh): 
    data = 1.98 * (tas - (0.55 - (0.0055 * rh)) * (tas - 58)) - 56.83
    return data

def algorithm_15(tas, wbt):
    data = (0.4 * wbt) + (0.6 * tas)
    return data

def algorithm_16(tas, wbt):
    data = (tas * 0.3) + (0.7 * wbt)
    return data

def algorithm_17(tas, es):
    data = ((tas * 0.567) + (0.393 * es) + 3.94)
    return data

def algorithm_18(tas, es):
    data = tas + (0.5555 * (es - 10.0))
    return data

# Create a registry of algorithms
@dataclass
class HeatStressAlgorithm:
    name: str
    func: Callable
    inputs: List[str]
    units: Dict[str, str]
    

algorithm_registry = {
    "heat_index_1_kalksteinvalimont1986_AT1":        HeatStressAlgorithm("algorithm_1", algorithm_1, ["tas", "td"], {"tas": "Celsius", "td": "Celsius"}),
    "heat_index_2_rothfusz1990_AT2":                 HeatStressAlgorithm("algorithm_2", algorithm_2, ["tas", "rh"], {"tas": "Fahrenheit", "rh": "%"}),
    "heat_index_3_AT3_NWS2025b":                     HeatStressAlgorithm("algorithm_3", algorithm_3, ["tas", "rh"], {"tas": "Fahrenheit", "rh": "%"}),
    "heat_index_4_smoyerrainham2001_AT4":            HeatStressAlgorithm("algorithm_4", algorithm_4, ["tas", "td"], {"tas": "Celsius", "td": "Celsius"}),
    "heat_index_5_blazejczyk2012_AT5":               HeatStressAlgorithm("algorithm_5", algorithm_5, ["tas", "rh"], {"tas": "Celsius", "rh": "%"}),
    "heat_index_6_steadman1984_AT6":                 HeatStressAlgorithm("algorithm_6", algorithm_6, ["tas", "es"], {"tas": "Celsius", "es": "kPa"}),
    "heat_index_7_schoen2005_AT7":                   HeatStressAlgorithm("algorithm_7", algorithm_7, ["tas", "td"], {"tas": "Celsius", "rh": "%"}),
    "heat_index_8_robinson2001_AT8":                 HeatStressAlgorithm("algorithm_8", algorithm_8, ["tas", "rh"], {"tas": "Fahrenheit", "rh": "%"}),
    "heat_index_9_stull2011_AT9":                    HeatStressAlgorithm("algorithm_9", algorithm_9, ["tas", "rh", "es"], {"tas": "Celsius", "rh": "%", "es": "kPa"}),
    "heat_index_10_thom1959_DI1":                    HeatStressAlgorithm("algorithm_10", algorithm_10, ["tas", "wbt"], {"tas": "Fahrenheit", "wbt": "Fahrenheit"}),
    "heat_index_11_usweather1959_DI2":               HeatStressAlgorithm("algorithm_11", algorithm_11, ["tas", "td"], {"tas": "Fahrenheit", "td": "Fahrenheit"}),
    "heat_index_12_sohar1962_DI3":                   HeatStressAlgorithm("algorithm_12", algorithm_12, ["tas", "wbt"], {"tas": "Fahrenheit", "wbt": "Fahrenheit"}),
    "heat_index_13_schlatter1987_DI4":               HeatStressAlgorithm("algorithm_13", algorithm_13, ["tas", "rh"], {"tas": "Fahrenheit", "rh": "%"}),
    "heat_index_14_pepi2000_DI5":                    HeatStressAlgorithm("algorithm_14", algorithm_14, ["tas", "rh"], {"tas": "Fahrenheit", "rh": "%"}),
    "heat_index_15_wallace2005_WBDT":                HeatStressAlgorithm("algorithm_15", algorithm_15, ["tas", "wbt"], {"tas": "Celsius", "wbt": "Celsius"}),
    "heat_index_16_yaglouminard1957_WBGT1":          HeatStressAlgorithm("algorithm_16", algorithm_16, ["tas", "wbt"], {"tas": "Celsius", "wbt": "Celsius"}),
    "heat_index_17_ABOM1984_WBGT2":                  HeatStressAlgorithm("algorithm_17", algorithm_17, ["tas", "es"], {"tas": "Celsius", "es": "kPa"}),
    "heat_index_18_mastertonrichardson1979_HUMIDEX": HeatStressAlgorithm("algorithm_18", algorithm_18, ["tas", "es"], {"tas": "Celsius", "es": "kPa"}),
}

def build_metric_cube(algorithm_registry, algorithm_name, input_cubes):
    """
    Build a metric cube using the specified algorithm, input cubes, and unit conversions.
    """
    entry = algorithm_registry[algorithm_name]
    
    prepared_data = []
    for var in entry.inputs:
        cube = input_cubes[var].copy()
        # Ensure the input cube is converted to the expected units
        if cube.units != entry.units[var]:
            cube.convert_units(entry.units[var])
        prepared_data.append(cube.data)  # Use the masked data directly

    # Call the algorithm function with the prepared data
    result_data = entry.func(*prepared_data)

    # Create the output cube
    output_cube = input_cubes["tas"].copy()
    output_cube.data = result_data
    
    # Explicitly set the units of the output cube to match the algorithm's output
    output_cube.units = entry.units["tas"]
    
    # Convert the output cube's units to Kelvin
    if output_cube.units != "Kelvin":
        output_cube.convert_units("Kelvin")
    output_cube.long_name = f"Metric from {entry.name.replace('_', ' ').title()}"

    # Preserve the original mask from the input cubes
    if hasattr(input_cubes["tas"].data, "mask"):
        output_cube.data = np.ma.array(output_cube.data, mask=input_cubes["tas"].data.mask)

    return output_cube

def generate_all_metrics(input_dir, output_dir):
    """
    Generate all metric cubes using the registered algorithms.

    Parameters
    ----------
    input_dir: str or Path
        Directory containing the input NetCDF files (e.g., tas_masked.nc).
    output_dir: str or Path
        Directory to save the output metric cubes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, entry in algorithm_registry.items():
        output_path = output_dir / f"{name}.nc"
        if output_path.exists():
            print(f"Skipping {name}: Output already exists at {output_path}")
            continue

        try:
            # Load required input cubes
            input_cubes = {}
            for var in entry.inputs:
                cube_path = input_dir / f"{var}_masked.nc" # it is assumed files are masked to the region of interest
                input_cubes[var] = iris.load_cube(str(cube_path))

                if not cube_path.exists():
                    raise FileNotFoundError(f"Input file for {var} not found: {cube_path}")
                input_cubes[var] = iris.load_cube(str(cube_path))
            metric_cube = build_metric_cube(algorithm_registry, name, input_cubes)

            # Save the result
            iris.save(metric_cube, str(output_path))
            print(f"Saved: {output_path}")

            # Delete the metric cube from memory
            del metric_cube
            gc.collect()  # Get garbage collection to free memory

        except Exception as e:
            print(f"Failed to process {name}: {e}")

        # Delete input cubes from memory
        del input_cubes
        gc.collect()  # Get garbage collection to free memory

generate_all_metrics(input_dir="path/to/input_data_directory", output_dir="path/to/output_data_directory")