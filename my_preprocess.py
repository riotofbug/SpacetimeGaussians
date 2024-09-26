import os
from pathlib import Path

import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse
import collections
import struct
from pathlib import Path

sys.path.append(".")
from thirdparty.gaussian_splatting.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,read_extrinsics_text,read_intrinsics_text
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
import sys

def move_images_to_input_folder(folder):
    inputimagefolder = os.path.join(folder, "input")
    if not os.path.exists(inputimagefolder):
        os.makedirs(inputimagefolder)

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            source_path = os.path.join(folder, filename)
            target_path = os.path.join(inputimagefolder, filename)

            # 如果目标文件已存在，选择跳过或覆盖
            if not os.path.exists(target_path):
                shutil.copy(source_path,target_path)
            else:
                print(f"File {target_path} already exists, skipping.")

def write_colmap(path, cameras, offset=0):
    projectfolder = Path(os.path.join(path , os.path.join("frames",f"{offset:04d}")))
    manualfolder = Path(os.path.join(projectfolder , "manual"))

    if Path(projectfolder, "manual").exists():
        shutil.rmtree(Path(projectfolder, "manual"))
    if Path(projectfolder, "images").exists():
        shutil.rmtree(Path(projectfolder, "images"))
    if Path(projectfolder, "distorted").exists():
        shutil.rmtree(Path(projectfolder, "distorted"))
    if Path(projectfolder, "input").exists():
        shutil.rmtree(Path(projectfolder, "input"))
    if Path(projectfolder, "sparse").exists():
        shutil.rmtree(Path(projectfolder, "sparse"))
    if Path(projectfolder, "stereo").exists():
        shutil.rmtree(Path(projectfolder, "stereo"))
    if Path(projectfolder, "tmp").exists():
        shutil.rmtree(Path(projectfolder, "tmp"))
    if Path(projectfolder, "input.db-shm").exists():
        Path(projectfolder, "input.db-shm").unlink()
    if Path(projectfolder, "input.db-wal").exists():
        Path(projectfolder, "input.db-wal").unlink()
    if Path(projectfolder, "run-colmap-geometric.sh").exists():
        Path(projectfolder, "run-colmap-geometric.sh").unlink()
    if Path(projectfolder, "run-colmap-photometric.sh").exists():
        Path(projectfolder, "run-colmap-photometric.sh").unlink()

    manualfolder.mkdir(parents=True, exist_ok=True)  

    savetxt =  Path(os.path.join(manualfolder , "images.txt"))
    savecamera =  Path(os.path.join(manualfolder , "cameras.txt"))
    savepoints =  Path(os.path.join(manualfolder , "points3D.txt"))

    imagetxtlist = []
    cameratxtlist = []

    db_file =  Path(os.path.join(projectfolder , "input.db"))
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()


    for cam in cameras:
        id = cam['id']
        filename = cam['filename']

        # intrinsics
        w = cam['w']
        h = cam['h']
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']
        param = cam['intr']

        # extrinsics
        colmapQ = cam['q']
        T = cam['t']

        # check that cx is almost w /2, idem for cy
        # assert abs(cx - w / 2) / cx < 0.10, f"cx is not close to w/2: {cx}, w: {w}"
        # assert abs(cy - h / 2) / cy < 0.10, f"cy is not close to h/2: {cy}, h: {h}"

        line = f"{id} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {id} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        params = np.array((fx , fy, cx, cy,))
        
        camera_id = db.add_camera(1, w, h, params)
        cameraline = f"{id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
        cameratxtlist.append(cameraline)
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=id)
        db.commit()
        
        '''
        camera_id = db.add_camera(4, w, h, param)
        cameraline = f"{id} OPENCV {w} {h} {' '.join(param.astype(str))} \n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=id)
        db.commit()
        '''
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file

def getcolmapsinglen3d(folder, offset):
    folder = os.path.join(folder , os.path.join("frames",f"{str(offset).zfill(4)}"))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    move_images_to_input_folder(folder)
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

def convertdynerftocolmapdb(path, cameraspath, offset=0, downscale=1):

    try:
        cameras_extrinsic_file = os.path.join(cameraspath, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(cameraspath, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(cameraspath, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(cameraspath, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    cameras = []
    for k,v in cam_extrinsics.items():

        if cam_intrinsics[v.camera_id].model == "SIMPLE_PINHOLE":
            fx = fy = cam_intrinsics[v.camera_id].params[0]
        else:
            fx = cam_intrinsics[v.camera_id].params[0]
            fy = cam_intrinsics[v.camera_id].params[1]

        camera = {
            'id':v.id,
            'filename': v.name,
            'w': cam_intrinsics[v.camera_id].width,
            'h': cam_intrinsics[v.camera_id].height,
            'fx': fx,
            'fy': fy,
            'cx':cam_intrinsics[v.camera_id].params[-2],
            'cy': cam_intrinsics[v.camera_id].params[-1],
            'q': v.qvec,
            't': v.tvec,
            'intr': cam_intrinsics[v.camera_id].params,
        }
        cameras.append(camera)


    write_colmap(path, cameras, offset)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--datapath", default="", type=str)
    parser.add_argument("--cameraspath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=250, type=int)
    parser.add_argument("--downscale", default=1, type=int)

    args = parser.parse_args()
    datapath = Path(args.datapath)
    cameraspath = Path(args.cameraspath)

    startframe = args.startframe
    endframe = args.endframe
    downscale = args.downscale

    print(f"params: startframe={startframe} - endframe={endframe} - downscale={downscale} - datapath={datapath}")

    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file 
    for offset in tqdm.tqdm(range(startframe, endframe), desc="convertdynerftocolmapdb"):
        convertdynerftocolmapdb(datapath, cameraspath, offset, downscale)

    for offset in range(startframe, endframe):
        getcolmapsinglen3d(datapath, offset)