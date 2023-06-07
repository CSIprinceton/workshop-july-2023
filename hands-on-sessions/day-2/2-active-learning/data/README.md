# Active learning and constructing the training dataset for the model


## Objectives

This tutorial will demonstrate the usage of the active learning protocol used to construct a suitable training dataset to obtain a converged Deep Potential (DP) model. 
The essential principles of the active learning protocol is based on ["Zhang et.al., Phys. Rev. Mater., 3, 023804"](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.023804).
Using the first DP model for Si created earlier, this tutorial will go over the iterative refinement of this model following the active learning protocol. Due to time constraints,
this tutorial only serves as a demonstration of the active learning protocol and the final DP model obtained would not be suitable for high-level production stage simulations.


## Outline

This tutorial will cover the following:
* Necessary files and scripts for running LAMMPS DPMD calculations
* Necessary files for training the models using DeepMD-kit
* General overview of the active learning protocol


## Prerequisites
It is assumed that the participant has attended the previous hands-on sessions leading up to the development of the first DP model for Si. This tutorial will only focus on the active learning
protocol for further refinement of the developed DP model for Si.


## Active learning
The active learning protocol involves three steps:
* Exploration
* Labeling
* Training

### Exploration

**Exploration** involves the sampling of the configuration space in an efficient manner using the current version of the DP model. This is typically done by DPMD simulations using an ensemble of trained
DP models at every iteration of the active learning process. During the course of the exploration step, an indicator is used to monitor the configurations explored on-the-fly and select those with low 
prediction accuracy. These selected configurations are sent to the **Labeling** step. In the tutorial here, we will be using the maximum deviation of atomic forces between four DP models as a reliable 
indicator to identify configurations with low prediction accuracy.

The portion of the LAMMPS input script ```start.lmp``` that specifies this is:


```pair_style      deepmd ../frozen_model_1_compressed.pb ../frozen_model_2_compressed.pb ../frozen_model_3_compressed.pb ../frozen_model_4_compressed.pb out_file md.out out_freq ${out_freq}```


The above line instructs LAMMPS to perform a DPMD simulation with ```frozen_model_1_compressed.pb``` as the DP model for representing the potential energy surface (PES), with the deviations in the virial 
and atomic forces computed with respect to all the four models. By default only the maximal, minimal and average model deviations are output to the ```md.out``` file.

A typical ```md.out``` file looks like this:


```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       8.427915e-03       5.381944e-04       4.014585e-03       6.426034e-02       1.066602e-02       2.992327e-02
          10       7.420156e-03       1.623721e-03       3.983475e-03       6.004617e-02       1.504808e-02       3.167037e-02
          20       1.447545e-02       2.874226e-03       8.273970e-03       9.241611e-02       2.048283e-02       3.700585e-02
          30       1.825509e-02       2.341124e-03       9.962326e-03       9.237947e-02       1.080203e-02       4.304728e-02
```

where the first column indicates the DPMD step, the next three columns provide the maximal, minimal and average model deviations of the virial and the last three columns the maximal, minimal and average 
model deviations in atomic forces.

It is standard practice to consider configurations with low prediction accuracy within a specified range of maximum deviation in atomic forces; for e.g. {0.1 to 0.8 eV/A}. This is done so that configurations
that have extremely poor representability are not included in the training dataset. In this tutorial, we instead consider all configurations explored during the DPMD simulation for **Labeling**.

A final important point to note in the **Exploration** step is the range of thermodynamic variables such as temperature and pressure which need to be considered. This is usually constrained by the
problem at hand for which the DP model is being developed. For example, if we are interested in the properties of liquid water at room temperature, then a typical range of temperatures from 273 K to 320 K and 
1 bar pressure would suffice. However, considering configurations spanning a much larger range of thermodynamic variables generally makes the DP model more robust. 

In the Si system in this tutorial, we consider liquid Si at 1700 K, at pressures corresponding to +/- 10,000 bar and 1 bar.


### Labeling
**Labeling** involves generating __ab-initio__ energies and forces for the selected configurations from the **Exploration** step. This can be done by high-level quantum chemistry, or density functional theory
(DFT) methods. The labeled configurations are then added to the existing training dataset, which is then used in the new iteration for **Training**.

### Training
**Training** fits the ever-increasing dataset to represent the PES efficiently and accurately. A collection of DP models that differ only in their initialization are used in the **Training** step. 
These models are then frozen and used to perform DPMD simulations in the next iteration where the model deviations are obtained in the **Exploration** step.


The active learning protocol is implemented over several iterations until a suitable DP model that accurately represents the PES is obtained. A common rule-of-thumb that is used to gauge the suitability 
of the DP model involves the model deviation in atomic forces falling below a pre-defined threshold over the course of a sufficiently long (e.g. 100 ps) DPMD simulation.











