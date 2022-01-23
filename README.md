# DLA-VPS: Deep Learning Assisted Visual Parameter Space Analysis of Large Scale Cosmological Simulations

## Introuction

The 3D GAN model which reconstruct the Nyx simulation data from simulation parameter space.

## Prerequisites
* Anaconda 4.10.1
* python 3.7.11
* tensorflow-gpu 2.3
* cuda 10.1
* cudnn 7.6.5
* tensorflow_addons-0.15.0

## Pipeline

### Data
* We used the data [Nyx](https://amrex-astro.github.io/Nyx/) which devlope by Lawrence Berkeley National Laboratory. 
* Simulaiotn Info: 
  * Input parameters: Omega_M, Omega_B and hubble.
  * Output quantities: density, temperature, rho_e, phi_grav, x-momentum, y-momentum and z-momentum. 
  * Resolution: 32^3 to 8192^3. (64^3 in our case.)
* We save the simulation data as [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) formate. Please contact us if you need it. 
### Training
* Step1: Modify the `Nyx_Reconstruction/config/utlis/config.py` of the training. The setting that most need to be changed are as follows:  
  * variable  : the quantity choosing. (density, temp, rho_e, phi_grav, xmom. ymom and zmom)
  * dataSetDir: the directory of the tfRecords.
  * logDir    : the directory to save the trained models (generator and discriminator).
  * epochs    : the number of the training epochs.
  * batchSize : the batch size during training.
* Step2: 
