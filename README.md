# MAE263F_TrevorOshiro
Repository for code and files written for the MAE263F class at UCLA. 

### Homework 1 Submission:
All code and report files are contained within the Homework1 file in this repository. The helper functions directory contains the necessary mathematical operations needed within the other problem scripts and is referenced from them. Each of the directories with plots at the end of the name are the directories where the problems for each script stores the outputteed plots for the deliverables. 

All of the scripts are executable as long as it is executed from within the Homework1 directory. Necessary changes to the parameters or methods can be made within the command line upon execution of the script when prompted to input. While inputs are taken, the physical and simulation parameters stated within the deliverable are also used as defaults. Specific execution instructions are given below:
- Problem1.py script: Takes in the radii of each sphere, the type of calculation method, and the time step used for each calculation method. All deformation, displacement, velocity, and angle plots are outputted to the Problem1_plots directory.
- Problem2.py script: Takes in the simulation time. All deformation, displacement, and velocity, plots are outputted to the Problem2_plots directory.
- Problem3.py script: Takes in the maximum force value used to when plotting Euler's theory versus the simulation results, the simulation duration, and the desired simulaiton timestep. All deformation, velocity, and other plots are outputted to the Problem3_plots directory. The script also outputs calculations from both the simulation and Euler's theory within the command prompt at 2000N and 20000N for the maximum displacement (referenced within the report file)

