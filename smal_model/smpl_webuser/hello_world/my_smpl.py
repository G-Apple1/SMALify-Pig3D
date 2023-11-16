'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''
import argparse
from ast import parse
from cmath import pi
from distutils.cmd import Command
import sys
# sys.path.append('D:\Liyixuan\Garment-Pattern-Generator-master\SMPL\smpl')
sys.path.append('E:\Garment-Pattern-Generator-master\SMPL\smpl')
from smpl_webuser.serialization import load_model,  save_model
import numpy as np


def get_command_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--theat', help="the pose paras", type = str, default = None)
  parser.add_argument('--beat', help="the shape paras", type = str, default = None)
  parser.add_argument('--target_path', help = "path to save", type = str, default = None)
  args = parser.parse_args()
  print(args)
  return args


if __name__ == "__main__":
  command_args = get_command_args()

  betas = {
    "XXL": [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "XL": [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "L": [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "M": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "S": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }

  ## Load SMPL model (here we load the female model)
  ## Make sure path is correct
  # m = load_model( 'D:\\Liyixuan\\Garment-Pattern-Generator-master\\SMPL\\smpl\\models\\basicModel_f_lbs_10_207_0_v1.0.0.pkl' )
  # uv_path = "D:\\Liyixuan\\Garment-Pattern-Generator-master\\SMPL\\smpl\\smpl_webuser\\hello_world\\uv.obj"
  m = load_model( 'E:\\Garment-Pattern-Generator-master\\SMPL\\smpl\\models\\basicModel_f_lbs_10_207_0_v1.0.0.pkl' )
  uv_path = "E:\\Garment-Pattern-Generator-master\\SMPL\\smpl\\smpl_webuser\\hello_world\\uv.obj"
  ## Assign random pose and shape parameters
  # A pose
  # m.pose[:] = np.random.rand(m.pose.size) * 0.0
  # m.pose[50] = -pi/4
  # m.pose[53] = pi/4
  # m.betas[:] = np.ones(m.betas.size) * 2.3
  for key, value in betas.items():

    m.betas[:] = value
    

    # if command_args.beat is not None:
    #   m.betas = command_args.beat

    # if command_args.theat is not None:
    #   m.pose = command_args.

    # m.betas[4] = -1.5
    # m.betas[5] = -3
    ## Write to an .obj file
    outmesh_path = command_args.target_path + '\\' + key + 'smpl.obj'
    out_path = command_args.target_path + '\\' + key + 'smpl.pkl'
    save_model(m, out_path)
    template = open(uv_path, 'r')
    with open( outmesh_path, 'w') as fp:
        for v in m.r:
          fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        while 1:
          l = template.readline()
          if not l:
            break
          fp.write(l)
        # for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        #     fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

  ## Print message
  print ('..Output mesh saved to: ', outmesh_path )
