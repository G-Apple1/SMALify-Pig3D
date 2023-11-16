import os
import pickle as pkl
import numpy as np
import config
from smal_model.smpl_webuser.serialization import load_model

def align_smal_template_to_symmetry_axis(v, sym_file):
    # These are the indexes of the points that are on the symmetry axis 对称轴135
    # I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]
    I = [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
     220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
     248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 271, 272, 289, 290, 301, 321, 323, 327,
     329, 337, 348, 350, 351, 352, 353, 356, 357, 358, 362, 364, 365, 367, 369, 370,375,  377, 1009, 1010, 1011, 1013, 1014, 1016, 1019, 1021, 1023, 1028, 1029, 1034,
    1037, 1038, 1040, 1041, 1044, 1052, 1053, 1054, 1068, 1069, 1070, 1071, 1072, 1078, 1082, 1083, 1089, 1090, 1092, 1988, 1989, 1990, 1993]
    # v = v - np.mean(v)
    # y = np.mean(v[I,1])
    # v[:,1] = v[:,1] - y
    v[I,1] = 0

    # symIdx = pkl.load(open(sym_path)) (3889,)
    with open(sym_file, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        symIdx = u.load()

    
    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    # v[left[symIdx]] = np.array([1,-1,1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    # import json
    # inds_dict = {"center_inds":center_inds.tolist(),
    #              "left_inds": left_inds.tolist(),
    #              "right_inds":right_inds.tolist()}

    # with open("/home/xucg/pig3d_net/data/pig_smal_data/pig_symmetry_inds.json",
    #           'w', encoding="utf-8") as f1:
    #     json.dump(inds_dict, f1, indent=2)

    # try:
    #     assert(len(left_inds) == len(right_inds))
    # except:
    #     import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds

# Legacy
def get_smal_template(model_name, data_name, shape_family_id=-1):
    model = load_model(model_name)
    nBetas = len(model.betas.r)

    with open(data_name, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # Select average zebra/horse
    # betas = data['cluster_means'][2][:nBetas]
    betas = data['cluster_means'][shape_family_id][:nBetas]
    model.betas[:] = betas

    if shape_family_id == -1:
        model.betas[:] = np.zeros_like(betas)

    v = model.r.copy()
    return v


