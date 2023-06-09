import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation

def eval_R_error(estimated_R: np.ndarray, ideal_R: np.ndarray) -> np.float_:
    error_rot = Rotation.from_matrix(estimated_R @ ideal_R.T)
    return linalg.norm(error_rot.as_rotvec())

def eval_T_error(estimated_T: np.ndarray, ideal_T: np.ndarray) -> np.float_:
    error_rot = estimated_T - ideal_T
    return linalg.norm(error_rot)

def get_gtR(file_path):
    gtR = []
    with open(file_path) as f:
        temp = []
        for i, line in enumerate(f.readlines()):
            if (i % 5 == 0) or ((i + 1) % 5 == 0): #1,5,6,10,11...行目
                continue
            else:
                line = line.split()
                line = line[:3]
                line = [float(n) for n in line]
                temp.append(line)
                if (i + 2) % 5 == 0: #2,7,12...行目
                    temp = np.array(temp)
                    gtR.append(temp)
                    temp = []
    return gtR

def get_gtT(file_path):
    gtT = []
    with open(file_path) as f:
        temp = []
        for i, line in enumerate(f.readlines()):
            if (i % 5 == 0) or ((i + 1) % 5 == 0): #1,5,6,10,11...行目
                continue
            else:
                line = line.split()
                line = line[3:]
                line = [float(n) for n in line]
                temp.append(line)
                if (i + 2) % 5 == 0: #2,7,12...行目
                    temp = np.array(temp)
                    gtT.append(temp)
                    temp = []
    return gtT
