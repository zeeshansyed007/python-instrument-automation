# Python Instrument Automation – Voltage Sweep (SIM Mode)

This project demonstrates Python-based instrument automation for hardware validation workflows.

## Overview
The script performs an automated voltage sweep using a simulated power supply and digital multimeter.  
At each step, it measures voltage and current, logs results, generates summary statistics, and produces plots.

The project is structured to support both:
- **SIM mode** (simulated instruments, no hardware required)
- **REAL mode** (future integration using PyVISA / SCPI with lab instruments)

## Features
- Automated voltage sweep execution
- Measurement logging to CSV
- Summary report generation
- Automated plot generation
- Instrument abstraction layer for future VISA/SCPI integration

## Technologies Used
- Python
- pandas
- numpy
- matplotlib
- PyVISA (architecture-ready)

## Typical Use Case
Board-level power validation, automated lab testing, and reproducible hardware measurement workflows.
