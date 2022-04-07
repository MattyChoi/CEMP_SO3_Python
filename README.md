# Robust Group Synchronization via Cycle-Edge Message Passing (CEMP)

## CEMP Implementation in Python3

Python3 implementation of the CEMP algorithm on the SO(3) group (rotation averaging problems). To see a more comprehensive explanation and a larger list of applications of the CEMP method in Matlab, check out the original author's [github repository](https://github.com/yunpeng-shi/CEMP).

See more details in
[Robust Group Synchronization via Cycle-Edge Message Passing](https://link.springer.com/content/pdf/10.1007/s10208-021-09532-w.pdf), Gilad Lerman and Yunpeng Shi, Foundations of Computational Mathematics, 2021.

For other possible usages of CEMP, see repo (https://github.com/yunpeng-shi/MPLS) and (https://github.com/yunpeng-shi/IRGCL).

## Demo

<img src="https://github.com/MattyChoi/CEMP_SO3_Python/blob/main/cemp_demo.png" width="500" height="400">

To run the code, you will need to install [git](https://git-scm.com/downloads) to clone this repository. In the terminal in your desired location, run the command
```
git clone https://github.com/MattyChoi/CEMP_SO3_Python.git
```
to clone the repository. Then, change the current directory using
```
cd CEMP_SO3_Python
```
to enter the folder containing the code. Then run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed. Then, change the directory to the [Examples](https://github.com/MattyChoi/CEMP_SO3_Python/tree/main/Examples) folder
```
cd Examples
```
and run one of the demo files.

## Dependencies
I used Python 3.9.7 for this project. The list of python packages used for this project is listed in the [requirements.txt](https://github.com/MattyChoi/CEMP_SO3_Python/blob/main/requirements.txt) file.