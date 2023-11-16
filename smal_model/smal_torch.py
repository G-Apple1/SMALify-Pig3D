"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from .smal_basics import align_smal_template_to_symmetry_axis, get_smal_template
import torch.nn as nn
import config

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


'''
J_regressor_prior: 关节回归矩阵的先验
J_regressor: 关节回归矩阵
f: 面（3个点的索引）
kintree_table: 关节树表
J: 关节位置
weight_prior: 蒙皮权重先验
weight： 蒙皮权重
vert_sym_idxs: 顶点索引
posedirs: 姿态矫正
bs_type: 姿态矫正蒙皮方式（lrotmin）
v_template: T pose顶点信息（基础模型）
shapedirs: 形状矫正 
bs_style: 形状矫正蒙皮方式（lbs）
'''


def caclulate_bone_lengths_from_J(J, parents):
    # NEW: calculate bone lengths:
    all_bone_lengths_list = []
    for i in range(1, parents.shape[0]):
        bone_vec = J[:, i] - J[:, parents[i]]
        bone_length = torch.sqrt(torch.sum(bone_vec ** 2, axis=1))  # 一个bs中骨架一对关节点的距离
        all_bone_lengths_list.append(bone_length)
    all_bone_lengths = torch.stack(all_bone_lengths_list)

    return all_bone_lengths  # .permute((1,0))

class SMAL(nn.Module):
    def __init__(self, device, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        # -- Load SMPL params --
        # with open(pkl_path, 'r') as f:
        #     dd = pkl.load(f)
           
        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()

        self.f = dd['f']
        '''(7774,3)三角网格数量'''
        self.faces = torch.from_numpy(self.f.astype(int)).to(device)

        # replaced logic in here (which requried SMPL library with L58-L68)
        '''（3889,3）基础模型'''
        v_template = get_smal_template(
            model_name=config.SMAL_FILE,
            data_name=config.SMAL_DATA_FILE,
            shape_family_id=-1)

        v_sym, self.left_inds, self.right_inds, self.center_inds = \
            align_smal_template_to_symmetry_axis(v_template, sym_file=config.SMAL_SYM_FILE)
        # Mean template vertices
        self.v_template = Variable(torch.Tensor(v_sym),requires_grad=False).to(device)

        # Size of mesh [Number of vertices, 3]
        self.size = [v_template.shape[0], 3]
        '''(3889,3,41) >> 41'''
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis
        '''(41,11667)'''
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T.copy()

        self.shapedirs = Variable(
            torch.Tensor(shapedir), requires_grad=False).to(device)

        # if shape_family_id != -1:
        #     with open(config.SMAL_DATA_FILE, 'rb') as f:
        #         u = pkl._Unpickler(f)
        #         u.encoding = 'latin1'
        #         data = u.load()
        #     # Select mean shape for quadruped type
        #     '''(5,41) >> (41,) 指定基礎動物模型'''
        #     # betas = data['cluster_means'][shape_family_id]#类别模型
        #     betas = np.zeros_like(data['cluster_means'][shape_family_id])
        #     '''(3889,3)'''
        #     v_template = v_template + np.matmul(betas[None,:], shapedir).reshape(
        #         -1, self.size[0], self.size[1])[0]

        # (35,3889)
        # Regressor for joint locations given shape 
        # self.J_regressor = Variable(
        #     torch.Tensor(dd['J_regressor'].T.todense()),
        #     requires_grad=False).to(device)

        self.J_regressor = Variable(
            torch.Tensor(dd['J_regressor'].T), requires_grad=False).to(device)###


        # Pose blend shape basis =306
        num_pose_basis = dd['posedirs'].shape[-1]
        
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.Tensor(posedirs), requires_grad=False).to(device)
        # (2,35)
        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)
        self.kintree_table = dd['kintree_table']

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])),
            requires_grad=False).to(device)


    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True, v_template=None):

        if True:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        # print("\ntheta: ",theta)

        
        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        if v_template is None:
            v_template = self.v_template

        # 1. Add shape blend shapes
        
        if nBetas > 0:#20
            if del_v is None:
                # print("\nbeta: ", beta)
                v_shaped = v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v 
        '''3889个顶点转为35个关节点'''
        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        #(1,35,3)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # all_bone_length = caclulate_bone_lengths_from_J(J,self.parents)
        # print(all_bone_length.shape, all_bone_length)
        # np.savetxt("/media/scau2311/A/xcg/barc_release/data/pig_smal_data/mean_pig_bone_lengths.txt",
        #            all_bone_length, fmt="%.18f")

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if len(theta.shape) == 4:
            Rs = theta
        else:# N x 35 x 3 x 3 用3x3的旋转矩阵表示的关节点的全局旋转矩阵
            # theta[0,0] = torch.zeros(1, 3)
            Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        
        # Ignore global rotation.
        # (1,306) 当前姿态和306个静止姿态的相对旋转值 34X9
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, 306])
        # print(pose_feature)
        #(1,3889,3)
        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),#混合变形计算
            [-1, self.size[0], self.size[1]]) + v_shaped
        #J_transformed=(1,35,3)3元素？
        #A=(1,35,4,4)
        #4. Get the global joint location  以0号节点为根节点，其他节点先对其的旋转角度，表示模型的全局旋转
        self.J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents)#, betas_logscale=betas_logscale


        # 5. Do skinning:
        num_batch = theta.shape[0]
        #(3889,35)蒙皮权重
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

        #(1,3889,4,4)
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])), [num_batch, -1, 4, 4])
        #(1,3889,4)加1列全为1
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        #(1,3889,4,1)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=beta.device)

        verts = verts + trans[:, None, :]
        # print("tran: ",trans)

        # Get joints:变换后的关节位置
        # joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        # joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        # joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        # joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
        joints = self.J_transformed

        joints = torch.cat([
            joints,
            verts[:, None, 257], # 35 nose
            verts[:, None, 237],  # 36 chin
            verts[:, None, 3700],  # 37 left ear tip
            verts[:, None, 1820],  # 38 right ear tip
            verts[:, None, 3816],  # 39 left eye
            verts[:, None, 1936],  # 40 right eye
            verts[:, None, 321],  # 41 throat
            ], dim = 1) 

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(figsize=[10, 8])
        # ax = plt.axes(projection="3d")
        # joints_x, joints_y, joints_z = joints.cpu()[0][:, 0].detach().numpy(), joints.cpu()[0][:, 1].detach().numpy(), joints.cpu()[0][:,2].detach().numpy()
        # verts_x, verts_y, verts_z = v_template[:, 0].cpu().detach().numpy(), v_template[:, 1].cpu().detach().numpy(),v_template[:, 2].cpu().detach().numpy()
        #
        # # ax.scatter3D(joints_x, joints_y, joints_z, s=50, c='red', label='3d')
        # ax.scatter3D(verts_x, verts_y, verts_z, s=10, c='blue', label='3d',alpha=0.5)
        # for i, j in enumerate(config.PIG_MODEL_JOINTS_NAME):
        #     ax.text3D(joints.cpu()[0][i][0].detach().numpy(), joints.cpu()[0][i][1].detach().numpy(),
        #               joints.cpu()[0][i][2].detach().numpy(), j)
        # # ax.scatter3D(proj_points[0][:, 0].detach().numpy(), proj_points[0][:, 1].detach().numpy(),
        # #              np.zeros_like(proj_points[0][:,0].detach().numpy()), s=50, c='blue', label='2d')
        # ax.legend()
        # # kintree_table = [[6, 7], [7, 8], [8, 11], [9, 10], [10, 11], [3, 4], [4, 5], [3, 4],
        # #                  [0, 1], [1, 2], [2, 5], [2, 8], [2, 15],
        # #                  [8, 16], [15, 19], [16, 20],
        # #                  [16, 22], [15, 21], [21, 17], [22, 17], [18, 17],
        # #                  [11, 12], [5, 12], [12, 13], [13, 14]]
        # for i in self.kintree_table.T:
        #     if i[0] > 35:
        #         i=[0,0]
        #     x1, y1, z1 = [], [], []
        #     x2, y2, z2 = [], [], []
        #     for j in i:  # 两个点相连
        #         x1.append(float(joints_x[j]))
        #         y1.append(float(joints_y[j]))
        #         z1.append(float(joints_z[j]))
        #         x2.append(float(Jx[0][j]))
        #         y2.append(float(Jy[0][j]))
        #         z2.append(float(Jz[0][j]))
        #     ax.plot3D(x1, y1, z1, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10,
        #               label="first")
        #     # ax.plot3D(x2, y2, z2, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10, label="second")
        # ax.text3D(x1[0], y1[0], z1[0], "3d", fontsize=10)
        # # ax.text3D(x2[0], y2[0], z2[0], "second", fontsize=10)
        # # plt.savefig(rf"E:\DL\SMALify\outputs\pigs\vis_joints\{time.time()}.png")
        # # plt.close('all')
        # plt.xlabel('X')
        # plt.ylabel('Y')  # y 轴名称旋转 38 度
        # ax.set_zlabel('Z', rotation=90)  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
        # import time
        # time0 = time.time()
        # # plt.savefig(f"/media/scau2311/A/xcg/SMALify/outputs/pigs/000000054901/vis_results/3d_joint_{time0}.jpg")
        # plt.pause(10)
        # plt.show()

        if get_skin:
            return verts, joints, Rs, v_shaped##
        else:
            return joints
