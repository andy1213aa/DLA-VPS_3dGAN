# DLA-VPS (3D GAN based model)

## Introuction

The 3D GAN model which reconstruct Nyx simulation data from the simulation parameter space.

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
  * Input parameters: Omega_M: [0.17, 0.5] , Omega_B: [0.03, 0.08] and hubble: [0.55, 0.85].
  * Output quantities: density, temperature, rho_e, phi_grav, x-momentum, y-momentum and z-momentum. 
  * Resolution: 32^3 to 8192^3. (64^3 in our case.)
* We save the simulation data as [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) formate. Please contact us if you need it. 
### Training

* Step1: Modify the `Nyx_Reconstruction/config/utlis/config.py` of the training. The settings that are most needed to be changed are as follows:  
  * variable  : the quantity choosing. (density, temp, rho_e, phi_grav, xmom. ymom and zmom)
  * dataSetDir: the directory of the tfRecords
  * logDir    : the directory to save the trained models (generator and discriminator)
  * epochs    : the number of the training epochs
  * batchSize : the batch size during training
  * save_weights_only: save wieghts only or save model. 
  * save_epochs: how many epochs to save the trained model once
  
* Step2: Run `main.py` to start the training. The trained models will be saved under the `logDir/gen` and `logDir/dis` with respective to the generator and discriminator during training. We highly recommend you to traine the models on GPU instead of CPU.
* Step3: Buy a coffee and have patience!

### Architecture

#### Generator
![generator](https://user-images.githubusercontent.com/59753286/150675090-f6e2ac97-0860-4357-bba6-addba81955d7.png)
#### Discriminator 
![discriminator](https://user-images.githubusercontent.com/59753286/150675104-713c5835-e5e0-4203-8a3a-94a68f5904d7.png)
#### Ressidual block
![residual_block](https://user-images.githubusercontent.com/59753286/150675117-8bea2bdb-5bd0-47bb-b77a-2249d1001b40.jpg)

### Notes
* The details of the model architecture and training process can be found in the theses [DLA-VPS](https://www.airitilibrary.com/Publication/alDetailedMesh1?DocID=U0021-NTNU40243)
