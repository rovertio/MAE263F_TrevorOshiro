# MAE263F_TrevorOshiro
Repository for code and files written for the MAE263F class at UCLA. 

### Final Project Submission
- The necessary files for submission can be found within the "Combined_Application" sub-directory of the "FinalProject" directory in this repository
- A separate READme.md file is within this directory with a procedure for executing the simulation and adjusting parameters for the simulation executions
- One would need to have the entire "FinalProject" directory on their computer to run the simulation, as helper functions are found within the "MMM_combine.py" file. 

### Homework 3 Submission:
Libraries needed: Ipython.display, numpy, matplotlib.pyplot
The scripts are split within the two directories within the Homework3 directory. The "HW3_LinearFit" directory contains the executable script Homework3.py that generates plots for the first deliverable of the homework. The "HW3_NonLinearFit" directory contains a Homework3.py script that is executable and generates plots for the second deliverable of the homework. When both of these scripts are run, the plots are stored within the "Plots" directory within the respective locations. 

The graphs for the problem can be obtained through the execution of the Homework3.py are obtained as follows:
- Upon execution of the script, values of learning rates and epoch numbers are iterated through and will show plots via a pop-up. Close the plots to finish execution of the program
- Different values for learning rates and epochs can be adjusted by changing the values of the iteration count and increment values within the code. Note that the for loop may increment via multiplication or addition, so comment out lines for the desired method of incrementing.


### Midterm Report/Presentation Submission:
All necessary files for the project progress as of the midterm can be found within the ProjectCodes directory under the MidtermConcept directory in this repository. The file for the presentation is the pdf file titled MidtermPresentation, and the file for the report is the pdf file titled as Midterm.

The code developed for this stage of the project for the Modified Mass Method implementation is found with the python file titled Bounce, and the python files containing helper functions within the HelperFunctions directory (within the MidtermConcept directory). The files are set to be run for one node upon execution of Bounce.py. Including more nodes for simulation of a straight beam is possible by uncommenting lines for variables f_n and J_n within the MMMadj.py file in the HelperFunctions directory. Lines initiating the node number and initial conditions would also need to be uncommented within the main function of the Bounce.py file.

Plots for the vertical components of displacement, velocity, reaction force, and z vector can be obtained when running the scripts. These plots are saved within the NodePlots folder of within the MidtermConcept directory. Note that these plots are only for the first node created within the script. 

For the program files related to the contact simulation via predictor/corrector methods, refer to the github repostory by Jessica Anz. All files related to this part of the project can be found within the Project_Dev directory in the repository.

### Homework 2 Submission:
Libraries needed: Ipython.display, numpy, mpl_toolkits.mplot3d, matplotlib.pyplot
The helper functions are all within the HelperFunctions folder that include computational functions needed within the main program file for the problem. 

The graphs for the problem can be obtained through the execution of the Homework2.py file in the Homework2 directory (executed within the Homework2 directory):
- Prompted to enter: a desired time step, number of nodes, and total simulation time to obtain plots (but the value defaults to those in the problem statement if nothing is entered into the prompt)
- Upon execution: show plots at certain time intervals of the simulation (user has to exit out of the plots to continue running the simulation)
- After the simulation ends: the plots at time steps and the plot of the end node coordinates will be saved into the Problem1_Plots directory in the Homework2 directory


### Project Proposal Submission (Team 13):
All files necessary for the assignment are in the ProjectProposal folder within this repository. This includes pdf documents of the proposal slides and the proposal report written. 


### Homework 1 Submission:
All code and report files are contained within the Homework1 file in this repository. The helper functions directory contains the necessary mathematical operations needed within the other problem scripts and is referenced from them. Each of the directories with plots at the end of the name are the directories where the problems for each script stores the outputteed plots for the deliverables. 

All of the scripts are executable as long as it is executed from within the Homework1 directory. Necessary changes to the parameters or methods can be made within the command line upon execution of the script when prompted to input. While inputs are taken, the physical and simulation parameters stated within the deliverable are also used as defaults. Necessary libraries are numpy and math for computations. Specific execution instructions are given below:
- Problem1.py script: Takes in the radii of each sphere, the type of calculation method, and the time step used for each calculation method. All deformation, displacement, velocity, and angle plots are outputted to the Problem1_plots directory.
- Problem2.py script: Takes in the simulation time. All deformation, displacement, and velocity, plots are outputted to the Problem2_plots directory.
- Problem3.py script: Takes in the maximum force value used to when plotting Euler's theory versus the simulation results, the simulation duration, and the desired simulaiton timestep. All deformation, velocity, and other plots are outputted to the Problem3_plots directory. The script also outputs calculations from both the simulation and Euler's theory within the command prompt at 2000N and 20000N for the maximum displacement (referenced within the report file)

