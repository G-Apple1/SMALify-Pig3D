'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the serialization functions of the SMPL model. 

Modules included:
- save_model:
  saves the SMPL model to a given file location as a .pkl file
- load_model:
  loads the SMPL model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''

__all__ = ['load_model', 'save_model']

import numpy as np
#import cPickle as pickle
#import _pickle as pickle
import pickle
import chumpy as ch
from chumpy.ch import MatVecMult
from .posemapper import posemap
from .verts import verts_core
    
def save_model(model, fname):
    m0 = model
    trainer_dict = {'v_template': np.asarray(m0.v_template),'J': np.asarray(m0.J),'weights': np.asarray(m0.weights),'kintree_table': m0.kintree_table,'f': m0.f, 'bs_type': m0.bs_type, 'posedirs': np.asarray(m0.posedirs)}    
    if hasattr(model, 'J_regressor'):
        trainer_dict['J_regressor'] = m0.J_regressor
    if hasattr(model, 'J_regressor_prior'):
        trainer_dict['J_regressor_prior'] = m0.J_regressor_prior
    if hasattr(model, 'weights_prior'):
        trainer_dict['weights_prior'] = m0.weights_prior
    if hasattr(model, 'shapedirs'):
        trainer_dict['shapedirs'] = m0.shapedirs
    if hasattr(model, 'vert_sym_idxs'):
        trainer_dict['vert_sym_idxs'] = m0.vert_sym_idxs
    if hasattr(model, 'bs_style'):
        trainer_dict['bs_style'] = model.bs_style
    else:
        trainer_dict['bs_style'] = 'lbs'
    pickle.dump(trainer_dict, open(fname, 'wb'), -1)


def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'



def ready_arguments(fname_or_dict):#write dd with V_posed('V_template + posedirs')

    if not isinstance(fname_or_dict, dict):
        #dd = pickle.load(open(fname_or_dict))
        dd = pickle.load(open(fname_or_dict,'rb'), encoding='iso-8859-1')
    else:
        dd = fname_or_dict
    #命名规范检查
    backwards_compatibility_replacements(dd)#replace the name of parameters
        
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3 # number of pose parameters

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1]) #init 'trans','pose','beta' to 0

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    dd["f"] = np.asarray(dd["f"], dtype=np.int)

    if want_shapemodel:
        #修正后的定点信息v_shaped = shapedirs（形状纠正） 点乘 betas（形状参数） + v_template（T pose 定点信息）
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']#get vert of shaped
        v_shaped = dd['v_shaped']
        #J_regressor 和 v_shaped 计算出形状矫正后的新关节位置 J
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:,0])        
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:,1])        
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:,2])      #regress J_pos from v_shaped
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        #v_posed的计算调用了其他脚本方法 posemap
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:    
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
            
    return dd



def load_model(fname_or_dict):
    dd = ready_arguments(fname_or_dict)##打开pkl文件 输入模板
    
    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'bs_style': dd['bs_style'],
        'want_Jtr': True,
        'xp': ch,
    }
    # 一个重载函数，在蒙皮的过程中模型空间和世界空间进行转换，返回值是顶点信息和Jtr
    # linear blend skining
    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1,3))
    result.J_transformed = Jtr + dd['trans'].reshape((1,3))

    for k, v in dd.items():
        setattr(result, k, v)
        
    return result

