# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 1 - July 11, 2023

Installation of DeePMD-kit, Quantum Espresso, and visualization software

## Aims

DeePMD-kit is a package written in Python/C++ that implements deep learning algoritms for molecular simulation.
In particular, it is able to learn the potential energy surface, dipole moments, and polarizability from appropriate training sets based on electronic structure calculations.
The aim of this tutorial is to describe the available installation methods for DeePMD-kit and other software (such asx Quantum Espresso, Plumed and Ovito) that will be used during the tutorial.

## Objectives

The objectives of this tutorial session are:
- Learn the available installation methods for DeePMD-kit and understand which one is appropriate for your laptop computer or HPC cluster.
- Illustrate the installation methods
- Describe common issues and how to solve them
- Install Quantum Espresso
- Install the visualization software Ovito and Xcrysden in order to use them in the tutorials
- Install Plumed

## Prerequisites

It is assumed that the student is familiar with the linux command line and previous experience with conda and software compilation in linux is recommended.

## Installation of DeePMD-kit

The installation methods are thoroughly discussed in the deepmd-kit [manual](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/index.html).
Here, we will discuss the different options in detail.
In most cases the easy install procedure based on the conda package manager is the best option and you can also use it in HPC facilities (clusters).

### Easy install 

#### Conda

This easy install procedure uses the conda package manager.
Anaconda is often available in computer clusters, sometimes through [Environment Modules](https://modules.readthedocs.io/en/latest/).
Assuming that conda is not installed, lets go through the installation steps for anaconda or miniconda.
Miniconda is a minimal installer and is therefore recommended.
You can obtain miniconda by downloading the appropriate file from this [website](https://docs.conda.io/en/latest/miniconda.html).
Assuming that you are using a linux command line in a standard x86-64 architecture, you can run:
```
chmod +x Miniconda3-latest-Linux-x86_64
./Miniconda3-latest-Linux-x86_64
```
and follow the instructions within that script.

Now that conda is installed, the deepmd-kit is simply installed with the command:
```
conda create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps -c https://conda.deepmodeling.com -c defaults
```
This command creates a conda environment ```deepmd``` and installs all the dependencies that are needed.
See the [manual](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html#install-with-conda) for alternatives and for a suitable command to install a GPU version.

You can then enable the environment by running:
```
conda activate deepmd
```
and test that the ```dp``` (DeePMD-kit) and the ```lmp``` (LAMMPS) executables are available. 
If that works, congratulations! You are ready to do molecular dynamics simulations driven by ab initio machine learning potentials and much more!

#### Docker

An alternative to conda is using a docker container.
You can find instructions to install docker [here](https://docs.docker.com/engine/install/).
In Ubuntu linux you can use:
```
sudo apt update
sudo apt install docker.io
```
Next you can get the image with:
```
docker pull ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
```
See also other available images in the DeePMD-kit [manual](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html#install-with-docker).
The docker image can be run with the command:
```
docker run -it ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
```
and you can test that the executables ```dp``` and ```lmp``` are available. 

### More complicated scenarios

#### Installation from scratch

In some situations, for instance when one needs to compile software in special computer architectures, the easy install procedure will not work because appropriate conda packages or a suitable docker image are not available.
In other instances, one might need optional LAMMPS packages that were not included in the conda package, or we may be interested in making modifications to the DeePMD-kit or LAMMPS source code.
In this case we will have no option but to compile from scratch.
We will have to get our hands dirty but it will be worthwhile! Patience and lots of coffee! (or your favorite caffeinated drink)

The instructions are described well in the DeePMD-kit [manual](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/install-from-source.html).
The main issue is that DeePMD-kit has tensorflow as a dependency and its compilation is non trivial and time consuming.

## Installation of visualization software

### Ovito

We will use the Ovito visualization software to analyze the simulations in other tutorial sessions during the workshop.
Lets install it in your laptop or desktop computer!
The procedure is quite simple and you can find it [here](https://www.ovito.org/).

If you are using **linux** you can download a tarball file using this [link](https://www.ovito.org/download/3106/).
Then untar the file, cd to the appropriate folder and run,
```
tar -xf ovito-basic-3.8.4-x86_64.tar.xz
cd ovito-basic-3.8.4-x86_64/bin
./ovito
```

In **Windows**, just download the executable [here](https://www.ovito.org/) and follow the instructions in your screen.

During the workshop we will give a brief overview of Ovito's usage.

### Xcrysden

We will use Xcrysden to visualize the input and ouput files of Quantum ESPRESSO.
The installation instructions can be found [here](http://www.xcrysden.org/Download.html).
In Ubuntu linux (or in general, in debian-based distributions) the instructions are as follows:
```
apt install tk libglu1-mesa libtogl2 libfftw3-3 libxmu6 imagemagick openbabel libgfortran5
wget http://www.xcrysden.org/download/xcrysden-1.6.2-linux_x86_64-shared.tar.gz
tar -xf xcrysden-1.6.2-linux_x86_64-shared.tar.gz
cd xcrysden-1.6.2-bin-shared/bin
./xcrys
```

## Installation of Quantum ESPRESSO

In the [hands-on session 2](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/2-quantum-espresso) we will illustrate basic concepts about DFT calculations using [Quantum ESPRESSO](https://www.quantum-espresso.org/). 
The basic installation steps for version 7.2 are,
```
git clone https://gitlab.com/QEF/q-e.git -b qe-7.2
cd q-e
./configure
make -j4 all
```

## Installation of Plumed

[Plumed](https://www.plumed.org/) is an enhanced sampling plugin which can be interfaces with many molecular dynamics engines.
The conda installation of DeePMD-kit includes Plumed and, if that installation path was followed, no further action is needed.
For more complex installation scenarios we suggest reading the [Plumed manual](https://www.plumed.org/doc-v2.8/user-doc/html/_installation.html) and the [LAMMPS manual](https://docs.lammps.org/Build_extras.html#plumed).


## Virtual Machine Instructions
### Transferring Files with `scp`
When using `scp`, you need to use `-P` to specify the port number:
```bash
scp -P PORT -r tmp deepmd23user@lab-REPLACE.eastus.cloudapp.azure.com:/home/deepmd23user 
scp -P PORT -r deepmd23user@lab-REPLACE.eastus.cloudapp.azure.com:/home/deepmd23user/tmp .
```
Remember to replace `PORT` and `REPLACE` with your own port number and azure lab link.

### Easy Login
You can use **sshkey** to simplify the login and uploading commands. It requires three steps:
1. Generating an ssh key:
```bash
ssh-keygen -t ed25519
```
When you see this
```
Generating public/private ed25519 key pair.
Enter file in which to save the key (/Users/yifan/.ssh/id_ed25519): 
```
type the target indentity file. For example, I use `/Users/yifan/.ssh/azuser`.
Enter a passphrase you like (it can be empty) to save the indentity file.

2. Modify the `config` file:
```
vi ~/.ssh/config
```
Go to the end of the file, and add the following lines:
```bash
Host azuser
  HostName lab-REPLACE.eastus.cloudapp.azure.com
  Port PORT
  IdentityFile PATH/.ssh/azuser
  User deepmd23user
```
Remember to replace `REPLACE`, `PORT`, and `PATH` with your own azure lab link, port number, and the path to your `.ssh` folder.

3. Copy the ssh id:
```bash
ssh-copy-id -i PATH/.ssh/azuser -p PORT deepmd23user@lab-REPLACE.eastus.cloudapp.azure.com
```
The same as before, use your own proper `PATH`, `PORT`, and `REPLACE`.
You will be prompted to enter the password of your azure lab virtual machine.

After successfully doing this step, you can use the following commands to log in or transfer files:
```bash
ssh azure
scp -r temp azure:/home/deepmd23user/
```

Here you go. Enjoy the simple commands!

### Troubleshooting
- Virtual machine cannot connect
Sometimes when you try to log into the virtual machine using `ssh`, the following error message may appear:
```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:y0Lgjr7d7lw+rQnVXUU7JknCq2JzYrUDcMib0CBRUlA.
Please contact your system administrator.
Add correct host key in /Users/yifan/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /Users/yifan/.ssh/known_hosts:92
Host key for [lab-29e723ef-f8c9-4641-9982-3966bdf511db.eastus.cloudapp.azure.com]:5006 has changed and you have requested strict checking.
Host key verification failed.
```

If you see this error, you can delete the last 3 lines (or sometime 2 lines) that starts with `[lab-....eastus.cloudapp.azure.com]:` (execute these commands line by line):
```
vi ~/.ssh/known_hosts
Shift+g
2k
3dd
:wq
Enter
```

- Connection lost when using `tmux`
From Microsoft, there is a bug in the azure product that causes the heartbeat to fail when the user is running tmux. When you are trying to use tmux, the connection will get lost. Therefore, do NOT use `tmux` on this virtual machine.