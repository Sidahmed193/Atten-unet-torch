# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:19:26 2023

@author: Sid Ahmed Hamdad
"""

import torch

def lookforcuda():
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available!")
    
        # Get device count
        device_count = torch.cuda.device_count()
        print("Number of CUDA devices:", device_count)
    
        # Iterate through devices and print properties
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            print(f"Device {i} properties:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory}")
            print(f"  CUDA capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    else:
        print("CUDA is not available.")

   
  