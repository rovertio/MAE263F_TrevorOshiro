# MAE263F_TrevorOshiro Final Project
This directory contains all the materials required for the final project submission in the MECH&AE263F class. 

## Script Execution:
- Libraries needed are: matplotlib and numpy
- Script should be excesuted within this directory

## Parameter Adjustment
- Obstacles: Within the "MMM_combine.py" python file, the "test_col" function should be ajusted to include custom obstacles. An approximate mathematic expression would be needed to generate conditions for colission and the calculations for the p vector (unit vector normal to the surface) within the Modified Mass method. The plotting functions within the "EndEffectorGrasp.py" script would also have to be adjusted fro proper visualization of the contact.
- Gripper Geometry: The "EndEffectorGrasp.py" would have to be adjusted. There is a section titled "Gripper prong geometry definition (displacement)" where the initial displacement values of the node can be adjusted to match the desired geometry. As such, an equation representing the curvature or shape would be helpful
- Physical properties: The "EndEffectorGrasp.py" would have to be adjusted. There is a section titled "Physical properties of gripper prong", where properties such as the cross-sectional area and elasticity can be modified.
- End-effector actuation: The "EndEffectorGrasp.py" would have to be adjusted. There is a section titled "Physical properties of gripper prong", where the external forces on the gripper prong can be adjusted. However, this script only accounts for constant acceleration as of this revision.

## Plot Generation
- The python file titled "EndEffectorGrasp.py" is to be executed from either a code editor or command line
- Plot generated from the script pop up periodically during the simulation. Close these plots to progress through.
- All plots generated within the scrip execution will be saved in the "CombineApplicationPlots" directory. Periodic plots of the gripper will be saved to the "Combine_GeomPlots" sub-directory, and plots showing contact will be saved to the "Combine_ContactPlots" sub-directory.
- Each file for the plot is in png format and labeled with the time step taken at. If a new simulation is to be run, one would have to either delete or move the current contents to avoid overwriting previous data.

## Example Plots Generated
More information regarding the results of the project can be found within the project report (the pdf file named "FinalReport") and the presentation of the project (the pdf file named "FinalPresentation"). Some example plots are shown below from the script:

### Plot of reaction forces at nodes:


![MagnitudeReactionForceNodes](https://github.com/user-attachments/assets/06d17dc7-b6ab-4c0a-acb8-a6d9bf0d647c)

### Plot of displacement of nodes:


![Geom0 02798999999999903](https://github.com/user-attachments/assets/9b7e47f0-8afa-4493-8be3-944026e294ab)

### Plot of contact between nodes and surface:


![ContactGripperGeom0 04034](https://github.com/user-attachments/assets/6d385d64-a5a3-4234-ba5d-5caac6554af6)
