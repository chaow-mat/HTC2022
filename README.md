# Code for Helsinki Tomography Challenge 2022
## Authors, institution, location
Below is the information for the authors.
 + *Author*       Chao Wang and Ji Li
 + *Institution*  Department of Statistics & Data Science and Department of Mathematics, National University of Singapore
 + *Location*    21 Lower Kent Ridge Rd, Singapore 119077  
 -------
## Brief description of your algorithm and a mention of the competition
 Our reconstruction algorithm is a deep learning approach, as there have been provided the training datasets. Our deblurring network backbone is a U-net. We use the supervised learning.
 
The given dataset is very limited. To address such issue, we proposed the following data augmentations:
 + To increase the dataset scale, we propose using the subsets of the full sinogram to construct several dataset.
 + To mitigate the learning difficulty, we plug the *FBP* recovery (using astra-toolbox) as the input, as well we observe that the circle is generally destroyed.
 + We test the unrolling network, it does not work for the limited dataset.

 This code repository is uploaded for competition of Helsinki Tomography Challenge 2022.

 ## Installation instructions, including any requirements
See the ```requirement.txt``` to install the dependent packages and libraries.

 + Clone the github repository
   
   ```python
   git clone https://github.com/chaow-mat/HTC2022.git
   cd HTC2022
   ```
  + Use ```virtualenv```  to construct the virtual environment
    ```python
    pip3 install virtualenv
    virtualenv --no-site-packages --python=python3 htc2022
    source htc2022/bin/activate # enter the environment
    pip3 install -r requirements.txt # install the dependency 
    # deactivate
    ```
 + Install the Cython module
    ```
    pip3 install Cython
    ```
 + Install Astra-toolbox 
    ```bash
    git clone https://github.com/astra-toolbox/astra-toolbox.git
    cd astra-toolbox/build/linux
    bash ./autogen.sh   # when building a git version
    bash ./configure --with-cuda=/usr/local/cuda \
                --with-python \
                --with-install-type=module
    make
    make install
    cd ~current working directory
    ```
 
 + (*Automatically download the trained models*) In our code, we use the `gdown` to obtain the training checkpoints in google driver. The following links is used in our evaluations.
   +  Files for seven models are readable by anyone at [Here](https://drive.google.com/drive/folders/1K2KwABjR8oL21MXseu1weJOXuT1ohE0V)
   + If downloading fails, please download them from the above links and put them in the folder `checkpoints`

 ## Usage instructions
 
 + For recovering the binary segmented image from sinogram data, using the following command line in terminal
   ```python
   CUDA_VISIBLE_DEVICES=0 python3 main.py /path/to/input/folder/ /path/to/output/folder/ groupNbr
   ```
  Here `groupNbr` is the difficulty level of the recovery task, its values are the integers from 1 to 7, 1 means that there are 90-degree projected sinogram data, 7 means that there are only 30-degree projected sinogram data
