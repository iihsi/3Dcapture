import datetime
import os
import pprint
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from utils_me import eval_R_error, eval_T_error, get_gtR, get_gtT

"""
読み込む視点番号の決定．

入力RGB画像 color_NNNNNN.png
入力depth画像 depth_NNNNNN.png
入力マスク画像（ある場合）　omask_NNNNNN.png
NNNNNNは 000000から始まる6桁の連番

下記は，その連番入力データから，順番に読み込む番号を決め,int型のframes_nums配列に読み込む視点番号を決めるプログラム
"""

# パラメータの定義を行う

R_error = True
T_error = True
save_log = True 

num_Frames = 110  # フォルダに入ってる多視点RGBD画像の視点数
skip_N_frames = 4  # 読み込むときのスキップ枚数指定．１の時は各視点読み込む
frames_nums = np.arange(0, num_Frames, skip_N_frames)

N = len(frames_nums)  # 視点数

"""
ICPアルゴリズムに関するパラメータ
    0:PointToPoint
    1:PointToPlane
    2:ForGeneralizedICP
"""
icp_num = 2
th = 0.05  # max_correspondence_distance

"""
SDFの生成に関するパラメータ
"""
voxel_length = 0.001  # meters # ~ 1cm
sdf_trunc = 0.01  # meters # ~ several voxel_lengths

"""
データの存在しているdirectoryを指定
"""
# data_dir= "TUM/Bunny/Synthetic_bunny_Circle/"
# data_dir= "TUM/Bunny/Synthetic_bunny_Wave/"

# data_dir= "TUM/Kenny/Synthetic_Kenny_Circle/"
# data_dir= "TUM/Kenny/Synthetic_Kenny_Wave/"

# data_dir= "TUM/Tank/Synthetic_Tank_Circle/"
# data_dir= "TUM/Tank/Synthetic_Tank_Wave/"

# data_dir= "TUM/Teddy/Synthetic_Teddy_Circle/"
# data_dir= "TUM/Teddy/Synthetic_Teddy_Wave/"

data_dir = "TUM/Leopard/Synthetic_Leopard_Circle/"
# data_dir= "TUM/Leopard/Synthetic_Leopard_Wave/"

# data_dir= "TUM/Teddy/Kinect_Teddy_Handheld/"
# data_dir= "TUM/Leopard/Kinect_Leopard_Turntable/"



"""
結果保存用ディレクトリの作成
"""

path_list = data_dir.split("/")
data = path_list[-2]
print(f"{data} used")

dTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not (os.path.isdir("Result")):
    os.makedirs("Result")
result_dir = f"Result/{dTime}_{data}/"
os.makedirs(result_dir)


# dataに合わせて正解データを読み込み
data_list = data.split("_")
if R_error:
    if "Synthetic" in data_list:
        if "Circle" in data_list:
            ideal_Rs = get_gtR("TUM/synthetic_circle_poses.txt")
        elif "Wave" in data_list:
            ideal_Rs = get_gtR("TUM/synthetic_wave_poses.txt")
    elif "Kinect" in data_list:
        ideal_Rs = get_gtR(data_dir + "markerboard_poses.txt") 
    elif "rgb_inhand" in path_list:
        R_error = False
        print("You are using rgb_inhand dataset so R_error is not available.")
    elif "stanford" in path_list:
        R_error = False
        print("You are using stanford dataset so R_error is not available.")


if T_error:
    if "Synthetic" in data_list:
        if "Circle" in data_list:
            ideal_Ts = get_gtT("../data/TUM/synthetic_circle_poses.txt")
        elif "Wave" in data_list:
            ideal_Ts = get_gtT("../data/TUM/synthetic_wave_poses.txt") 
    elif "Kinect" in data_list:
        ideal_Ts = get_gtT(data_dir + "markerboard_poses.txt") 
    elif "rgb_inhand" in path_list:
        T_error = False
        print("You are using rgb_inhand dataset so T_error is not available.")
    elif "stanford" in path_list:
        T_error = False
        print("You are using stanford dataset so T_error is not available.")



# dataに合わせてuse_omaskを設定
if ("Synthetic" in data_list) or ("stanford" in path_list):
    use_omask = 0
else:
    use_omask = 1  # TUMのKinectデータ, rgb-inhandデータを使うときは1にする

############ カメラの内部パラメータの読み込みと設定#######################


class intrinsic:
    # RGBD画像群のディレクトリにある camera_params.txtの設定データから読み込む
    calib_file = data_dir + "camera_params.txt"
    with open(calib_file) as f:
        param_list = f.read().split("\n")
    image_width = int(param_list[0])
    image_height = int(param_list[1])
    fx = float(param_list[2])
    fy = float(param_list[3])
    cx = float(param_list[4])
    cy = float(param_list[5])


pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsic.image_width,
    intrinsic.image_height,
    intrinsic.fx,
    intrinsic.fy,
    intrinsic.cx,
    intrinsic.cy,
)

kdt = o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.05, max_nn=30
)  # 生成した３D点群の点毎に形状の法線ベクトルを求める関数のパラメータ設定

############ RGB画像とDepth画像を読み込む#######################

pcd = []  # PointCloudのリスト
rgbds = []  # 　rgbd画像のリスト

for i in range(0, N):  # 視点0~N-1まで

    # 入力のRGB画像のファイル名の設定
    rgb_file = data_dir + "color_%06d.png" % (frames_nums[i])
    # 入力のDepth画像のファイル名の設定
    depth_file = data_dir + "depth_%06d.png" % (frames_nums[i])

    # 3D点群（各視点）を保存するためのファイル名の設定
    ply_file = result_dir + "points_%06d.ply" % (frames_nums[i])
    all_ply_file = result_dir + "allpoints.ply"  # 全視点を統合した3D点群
    mesh_file = result_dir + "mesh.ply"  # 統合後生成する3Dメッシュデータ

    RGBim = cv2.imread(rgb_file, cv2.IMREAD_COLOR)  # RGB画像をファイルから読み込む
    RGBim = cv2.cvtColor(RGBim, cv2.COLOR_BGR2RGB)
    Depthim = cv2.imread(
        depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    )  # Depth画像をファイルから読み込む

    if RGBim is None or Depthim is None:
        print(
            "Could not load images: please check your path is correctly set",
            file=sys.stderr,
        )
        exit("LOAD_IMAGE_FAILURE")

    ###############################################################################
    if use_omask:
        # 物体マスク画像を利用してマスク画像の画素値が０の場合はそのDepth値を０としている．
        # 入力の物体領域マスク画像のファイル名の設定
        omask_file = data_dir + "omask_%06d.png" % (frames_nums[i])
        omask_image = cv2.imread(omask_file, cv2.IMREAD_GRAYSCALE)

        Depthim = (omask_image > 0) * Depthim
    ###############################################################################

    # 読み込んだRGB画像とDepth画像から，点群データを生成する

    color = o3d.geometry.Image(RGBim)  # RGB画像をOpen３D画像としてcolor画像に入力
    depth = o3d.geometry.Image(Depthim)  # Depth画像をOpen３D画像としてdepth画像に入力

    # color とdepthからOpen３Dのrgbd画像を生成
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    rgbds.append(rgbd)

    # 　生成したｒｇｂｄ画像をplot画面に表示
    plt.subplot(1, 2, 1)
    plt.title("color%d" % (frames_nums[i]))
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title("depth%d" % (frames_nums[i]))
    plt.imshow(rgbd.depth, cmap="gray")
    plt.show()

    # 生成したrgbd画像とカメラ内部パラメータから３Ｄ点群を生成
    p = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    # 生成した３D点群の各点群の法線ベクトルを計算
    o3d.geometry.PointCloud.estimate_normals(p, search_param=kdt)

    pcd.append(p) 

    o3d.io.write_point_cloud(ply_file, p)

    # セーブ済みのplyファイルから読み込む場合
    # p= o3d.io.read_point_cloud(ply_file)
    # pcd.append(p)

# ICP位置合わせ前の全３D点群を表示
o3d.visualization.draw_geometries(pcd, "input ", 640, 480)

############ 隣り合う視点のPointCloud間のICP位置合わせ#######################

RT = []  # i+1 番目のPointCloudをi番目のPointCloudに変換するRTのリスト
info_list = []  # ICPの評価結果を保存しておくリスト

RT.append(np.identity(4))

if icp_num == 0:
    icp = "PointToPoint"
elif icp_num == 1:
    icp = "PointToPlane"
elif icp_num == 2:
    icp = "ForGeneralizedICP"

#print(f"{icp} used")

for i in range(0, N - 1):
    #print("\nICP registration for %dth and %dth 3D point clouds" % (i + 1, i))

    T = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    if icp == "PointToPoint":
        info = o3d.pipelines.registration.registration_icp(
            pcd[i + 1],
            pcd[i],
            max_correspondence_distance=th,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
    elif icp == "PointToPlane":
        info = o3d.pipelines.registration.registration_icp(
            pcd[i + 1],
            pcd[i],
            max_correspondence_distance=th,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    elif icp == "ForGeneralizedICP":
        info = o3d.pipelines.registration.registration_generalized_icp(
            pcd[i + 1],
            pcd[i],
            max_correspondence_distance=th,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        )

    #print("correspondences:", len(info.correspondence_set))
    #print("fitness: ", info.fitness)
    #print("RMSE: ", info.inlier_rmse)
    #print("transformation: \n", info.transformation, flush=True)
    info_list.append(info)
    RT.append(info.transformation)

# 視点iを最初の視点（0）に変換するRT[i]を計算する．
for i in range(1, N):
    RT[i] = np.dot(RT[i - 1], RT[i])

# 各視点のPointCloud pcd[i]を最初の視点（0）に変換
for i in range(0, N):
    pcd[i].transform(RT[i])

# ICP位置合わせ後の全３D点群を表示
o3d.visualization.draw_geometries(pcd, "output ", 640, 480)

pcd_combined = o3d.geometry.PointCloud()
for i in range(0, N):
    pcd_combined += pcd[i]
o3d.io.write_point_cloud(all_ply_file, pcd_combined)

# ==============================================================================
##=============##         TSDF VOLUME INTEGRATION             ##=============##
# ==============================================================================

print("\n== TSDF volume integration: ==")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,  # meters # ~ 1cm
    sdf_trunc=sdf_trunc,  # meters # ~ several voxel_lengths
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

for i in range(0, N):
    volume.integrate(rgbds[i], pinhole_camera_intrinsic, np.linalg.inv(RT[i]))

# ==============================================================================
##=============##              TRIANGULAR MESH                ##=============##
# ==============================================================================

print(
    "\n== Extract a triangle mesh from the volume (with the marching cubes algorithm) and visualize it : ==",
    flush=True,
)

mesh = volume.extract_triangle_mesh()
print(mesh.compute_vertex_normals(), flush=True)

o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh(mesh_file, mesh)

prmdic = {}
prmdic["dataset"] = data
prmdic["num_Frames"] = num_Frames
prmdic["skip_N_frames"] = skip_N_frames
prmdic["icp"] = icp
prmdic["icp_max_correspondence_distance"] = th
prmdic["sdf_voxel_length"] = voxel_length
prmdic["sdf_sdf_trunc"] = sdf_trunc

print("Parameters")
pprint.pprint(prmdic)


"""
パラメータと行列の値をそれぞれ別のtxtファイルに出力する.
"""
if save_log:
    with open(f"{result_dir}prms.txt", "w") as f:  # 書き込みモードで開く
        for key, value in prmdic.items():
            f.write(f"{key}: {value}\n")

    with open(f"{result_dir}info.txt", "w") as f:
        for info in info_list:
            cor_len = len(info.correspondence_set)
            translist = info.transformation.tolist()
            f.write(f"correspondences: {cor_len}\n")
            f.write(f"fitness: {info.fitness}\n")
            f.write(f"RMSE: {info.inlier_rmse}\n")
            f.write("transformation:\n")
            for trans in translist:
                f.write(f"{trans}\n")
            f.write("\n")
    print("Log saved")

if R_error:
    estimated_Rs = []
    for rt in RT:
        estimated_Rs.append(rt[:3, :3])

    # frames_numsに対応する正解データを取得
    ideal_Rs = [ideal_Rs[fn] for fn in frames_nums]

    errors = []
    for estimated_R, ideal_R in zip(estimated_Rs, ideal_Rs):
        error = eval_R_error(estimated_R, ideal_R)
        errors.append(error)

    # errorをtxtファイルで保存する
    with open(f"{result_dir}error.txt", "w") as f:
        for i, error in zip(frames_nums, errors):
            # print(f"error[{i}]: {error}")
            f.write(f"error[{i}]: {error}\n")

    # errorをcsvファイルで保存する
    with open(result_dir + "errors.csv", "w") as f:
        for i, error in zip(frames_nums, errors):
            f.write(f"{i},{error}\n")

    # errorをPlotしたグラフを描写する
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100, facecolor="w")
    ax = fig.add_subplot(111, title="R_error", xlabel="R[i]", ylabel="R_error")
    ax.plot(frames_nums, errors)
    ax.set_xticks(frames_nums)
    plt.savefig(f"{result_dir}plot.pdf", dpi=300, bbox_inches="tight")


if T_error:
    estimated_Ts = []
    for rt in RT:
        estimated_Ts.append(rt[:3, 3:])

    # frames_numsに対応する正解データを取得
    ideal_Ts = [ideal_Ts[fn] for fn in frames_nums]

    errors = []
    for estimated_T, ideal_T in zip(estimated_Ts, ideal_Ts):
        error = eval_T_error(estimated_T, ideal_T)
        errors.append(error)

    # errorをtxtファイルで保存する
    with open(f"{result_dir}error.txt", "w") as f:
        for i, error in zip(frames_nums, errors):
            print(f"error[{i}]: {error}")
            f.write(f"error[{i}]: {error}\n")

    # errorをcsvファイルで保存する
    with open(result_dir + "errors.csv", "w") as f:
        for i, error in zip(frames_nums, errors):
            f.write(f"{i},{error}\n")

    # errorをPlotしたグラフを描写する
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100, facecolor="w")
    ax = fig.add_subplot(111, title="T_error", xlabel="T[i]", ylabel="T_error")
    ax.plot(frames_nums, errors)
    ax.set_xticks(frames_nums)
    plt.savefig(f"{result_dir}plot.pdf", dpi=300, bbox_inches="tight")

