'''
About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model 加载模型pkl文件
  - Edit pose & shape parameters of the model to create a new body in a new pose 对3d形状进行渲染
  - Save the resulting body as a mesh in .OBJ format 保存为obj文件

'''
import os.path
from cmath import pi
import sys
# sys.path.append('E:\Garment-Pattern-Generator-master\SMPL\smpl')

from smpl_webuser.serialization import load_model, save_model
import numpy as np

## Load SMPL model (here we load the female model)
## Make sure path is correct
# smpl_input =  "/media/scau2311/A/xcg/smalr_online/smpl_models/smal_00781_4_all.pkl"
# smpl_input = "/media/scau2311/A/xcg/smal-fitter/animated_fits_outputs/meshes/iterations/fits_outputs_pig_1999.pkl"
m = load_model( smpl_input )

## Assign random pose and shape parameters
# m.pose[:] = np.random.rand(m.pose.size) * 0.0
# m.pose[50] = -pi/4
# m.pose[53] = pi/4
# m.betas[:] = np.ones(m.betas.size) * 2.3


# m.betas[4] = -1.5
# m.betas[5] = -3
## Write to an .obj file
outmesh_path = os.path.dirname(smpl_input) + "/pkl_to_obj/" + os.path.basename(smpl_input)[:-4] + ".obj"
out_path = os.path.dirname(smpl_input) + "/pkl_to_obj/" + os.path.basename(smpl_input)[:-4] + ".pkl"
save_model(m, out_path)
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print ('..Output mesh saved to: ', outmesh_path )
