import os
import json
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import re
from archs.psrt_recurrent_arch import BasicRecurrentSwin
from basicsr.utils import tensor2img
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from tqdm import tqdm



def setup_paths_and_params(args):
    """Set up paths and parameters based on the scene type."""
    if not os.path.exists(args.source_path):
        os.makedirs(args.source_path)

    spynet_path = args.spynet_path if args.spynet_path is not None else f"vsr/{args.vsr_model}/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth"
    model_path = args.vsr_model_path if args.vsr_model_path is not None else f"vsr/{args.vsr_model}/experiments/pretrained_models/PSRT_Vimeo.pth"

    if os.path.exists(os.path.join(args.hr_source_path, "sparse")):
        # Colmap Dataset
        lr_trainset_path = os.path.join(args.lr_source_path, "images_down")
        transform_path = lr_trainset_path
        vsr_trainset_path = os.path.join(args.source_path, "images_vsr")
        # video_save_path = None
        # num_images_in_sequence = 10
        # similarity = 'pose'
        thres_values = args.thres_values + [ len(os.listdir(lr_trainset_path)), ]

        shutil.copytree(os.path.join(args.hr_source_path, "sparse"), os.path.join(args.source_path, "sparse"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(args.lr_source_path, "images_gt"), os.path.join(args.source_path, "images_gt"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(args.lr_source_path, "images_down"), os.path.join(args.source_path, "images_down"), dirs_exist_ok=True)
        shutil.copy2(os.path.join(args.hr_source_path, "poses_bounds.npy"), os.path.join(args.source_path, "poses_bounds.npy"))

    elif os.path.exists(os.path.join(args.hr_source_path, "transforms_train.json")):
        # Blender Dataset
        lr_trainset_path = os.path.join(args.lr_source_path, "train")
        transform_path = os.path.join(args.hr_source_path, "transforms_train.json")
        vsr_trainset_path = os.path.join(args.source_path, "train")
        thres_values = args.thres_values

        shutil.copytree(os.path.join(args.hr_source_path, "test"), os.path.join(args.source_path, "test"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(args.lr_source_path, "train"), os.path.join(args.source_path, "train_lr"), dirs_exist_ok=True)
        shutil.copy2(os.path.join(args.hr_source_path, "transforms_train.json"), os.path.join(args.source_path, "transforms_train.json"))
        shutil.copy2(os.path.join(args.hr_source_path, "transforms_val.json"), os.path.join(args.source_path, "transforms_val.json"))
        shutil.copy2(os.path.join(args.hr_source_path, "transforms_test.json"), os.path.join(args.source_path, "transforms_test.json"))
    else:
        raise ValueError("Could not recognize scene type!")

    video_save_path = args.video_save_path
    num_images_in_sequence = args.num_images_in_sequence
    similarity = args.similarity

    return spynet_path, model_path, lr_trainset_path, transform_path, vsr_trainset_path, video_save_path, num_images_in_sequence, similarity, thres_values



def load_vsr_model(
    spynet_path="psrt/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth", 
    model_path="psrt/experiments/pretrained_models/PSRT_Vimeo.pth", 
    device='cuda'
    ):
    """Load the VSR model with pre-trained weights."""
    model = BasicRecurrentSwin(
        mid_channels=64,
        embed_dim=120,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        window_size=[3, 8, 8],
        num_frames=3,
        cpu_cache_length=100,
        is_low_res_input=True,
        spynet_path=spynet_path
    )
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'], strict=False)
    model.eval()
    return model.to(device)


def extract_number(filename):
    s = re.findall("\d+", filename)
    return int(''.join(s)) if s else -1


def load_images(folder_path):
    images = []
    images_names = []
    
    filenames = [
        filename for filename in os.listdir(folder_path) 
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG')) 
        and 'normal' not in filename 
        and 'depth' not in filename
    ]
    filenames.sort(key=extract_number)

    for filename in filenames:
        images_names.append(filename)
        
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        images.append(image)

    return images, images_names


def read_extrinsics(input_path):
    dir_path = os.path.dirname(input_path)
    cameras_extrinsic_file = os.path.join(dir_path, "sparse/0", "images.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

    return cam_extrinsics


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles.
    
    Args:
        R (numpy.ndarray): 3x3 rotation matrix.
        
    Returns:
        tuple: A tuple containing (azimuth, elevation, roll).
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(z), np.degrees(y), np.degrees(x)


def calculate_elevation_azimuth(point):
    """
    Calculate the elevation and azimuth of a point in 3D space relative to the origin.
    
    Args:
        point (tuple or list or numpy.ndarray): A 3D point (x, y, z).
        
    Returns:
        tuple: A tuple containing (elevation, azimuth) in degrees.
    """
    x, y, z = point
    
    # Azimuth is the angle in the xy plane from the x-axis
    azimuth = np.arctan2(y, x)
    
    # Elevation is the angle from the xy plane upwards to the z axis
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    
    # Convert radians to degrees
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)
    
    return elevation_deg, azimuth_deg


def calculate_angle_between_points(point1, point2):
    """
    Calculate the angle between two points transformed by two different 4x4 matrices
    with respect to the origin.
    
    Args:
        matrix1 (numpy.ndarray): The first 4x4 transform matrix.
        matrix2 (numpy.ndarray): The second 4x4 transform matrix.
        
    Returns:
        float: The angle between the two points in degrees.
    """
    # Calculate the vectors from the origin to each point
    vector1 = point1
    vector2 = point2
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Ensure the cosine value is within the valid range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg < 0:
        import pdb ; pdb.set_trace()
    return angle_deg


def calculate_azimuth_difference(azimuth1, azimuth2):
    diff = np.abs(azimuth1 - azimuth2)
    return min(diff, 360 - diff)


def compute_features(images):
    orb = cv2.ORB_create()
    features = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        features.append((kp, des))
    return features

def compute_similarity(features1, features2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if features1[1] is None or features2[1] is None:
        return np.inf
    matches = bf.match(features1[1], features2[1])
    matches = sorted(matches, key=lambda x: x.distance)
    if matches:
        similarity = sum([m.distance for m in matches]) / len(matches)
    else:
        similarity = np.inf
    return similarity


def compute_position_distance_ranking(images, transform_path):
    camera_positions = {}
    if transform_path.endswith('.json'):
        with open(transform_path, 'r') as file:
            json_data = json.load(file)
        for frame in json_data['frames']:
            index = int(os.path.basename(frame['file_path']).split('_')[-1])
            transform_matrix = np.array(frame['transform_matrix'])
            camera_position = transform_matrix[:3, 3]  # Extract the camera position from the transformation matrix
            camera_positions[index] = camera_position
    else:
        cam_extrinsics = read_extrinsics(transform_path)
        for i in range(len(cam_extrinsics)):
            R = np.transpose(qvec2rotmat(cam_extrinsics[i+1].qvec))
            T = np.array(cam_extrinsics[i+1].tvec)
            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1) #.cuda()
            camera_center = world_view_transform.inverse()[3, :3].numpy()
            camera_positions[i] = camera_center

    distance_rankings = []

    for i in range(len(images)):
        distances = []
        elevation_i, azimuth_i = calculate_elevation_azimuth(camera_positions[i])
        for j in range(len(images)):
            if i == j:
                # distances.append([0, 0, 0, 0]) ### distance, azimuth diff, angle , elevation
                distances.append([0]) ### angle 
            else:
                # dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
                # elevation_j, azimuth_j = calculate_elevation_azimuth(camera_positions[j])
                # azimuth_diff = calculate_azimuth_difference(azimuth_i, azimuth_j)
                angle_diff = np.abs(calculate_angle_between_points(camera_positions[i], camera_positions[j]))
                distances.append([angle_diff])
                # distances.append([dist, azimuth_diff, angle_diff, elevation_j])
        
        sorted_indices = np.argsort(np.array(distances)[:, 0])
        rankings = [[np.where(sorted_indices == j)[0][0]] + distances[j] for j in range(len(images))]
        distance_rankings.append(rankings)

    return distance_rankings    # [ranking, distance, azimuth diff, angle , elevation]


def compute_feature_distance_ranking(images):
    features = compute_features(images)
    feature_distance_rankings = []

    for i in range(len(images)):
        distances = []
        for j in range(len(images)):
            if i == j:
                distances.append(0)
            else:
                dist = compute_similarity(features[i], features[j])
                distances.append(dist)
        
        sorted_indices = np.argsort(distances)
        rankings = [[np.where(sorted_indices == j)[0][0], distances[j]] for j in range(len(images))]
        feature_distance_rankings.append(rankings)

    return feature_distance_rankings



def ordering_sim1_thresholding_sim2(images, reference_index, inverse_threshold, similarity_1_rankings, similarity_2_rankings, simialrity_1_type):
    selected_indices = [reference_index]
    current_index = reference_index
    # Copy to avoid modifying the original array
    similarity_1_rankings_copy = similarity_1_rankings.copy()

    while len(selected_indices) < len(images):
        # Ordering
        similarity_1_rankings_copy[:, current_index, 0] = len(similarity_1_rankings_copy)

        for i in range(len(images)):
            candidate = np.argsort(similarity_1_rankings_copy[current_index][:, 0])[i]
            if candidate not in selected_indices:
                break

        # Trhesholding
        if similarity_2_rankings is not None:
            if simialrity_1_type == "feature":  # ordering : feature / thresholding : pose
                difference = similarity_2_rankings[current_index][candidate, 1]     # pose threshold : angle
            else:                               # ordering : pose    / thresholding : feature
                difference = similarity_2_rankings[current_index][candidate, 0]     # pose threshold : rank

            if difference > inverse_threshold:  # sim < epsilon
                break

        # Add the best candidate to selected_indices
        selected_indices.append(candidate)
        current_index = candidate

    return selected_indices


def process_S(
        model_vsr, similarity, images, names,
        save_path, transform_path,
        num_images_in_sequence, device='cuda'
    ):
    created_images = set()
    os.makedirs(save_path, exist_ok=True)

    all_sorted_image_paths = []
    total_outputs = []

    if similarity == 'feature':
        feature_rankings = np.array(compute_feature_distance_ranking(images))
        sorted_indices = ordering_sim1_thresholding_sim2(images, 0, 180, feature_rankings, None, similarity)
    elif similarity == 'pose':
        position_rankings = np.array(compute_position_distance_ranking(images, transform_path))
        sorted_indices = ordering_sim1_thresholding_sim2(images, 0, 180, position_rankings, None, similarity)
    # print(sorted_indices)

    # Inference VSR and save images
    for i in range(0, len(sorted_indices), num_images_in_sequence):
        chunk_indices = sorted_indices[i:i + num_images_in_sequence]

        # Handle case where the last chunk is smaller
        if len(chunk_indices) < num_images_in_sequence:
            chunk_indices = sorted_indices[-num_images_in_sequence:]

        # Prepare batch of images with context (like in code2)
        batch_imgs = [images[idx] for idx in chunk_indices]
        batch_imgs = [batch_imgs[0]] + batch_imgs + [batch_imgs[-1]]
        batch_imgs = np.stack(batch_imgs).transpose(0, 3, 1, 2)  # Convert to NCHW format
        batch_imgs = torch.from_numpy(batch_imgs).float().div(255.0).unsqueeze(0).to(device)

        # Generate outputs
        with torch.no_grad():
            outputs = model_vsr(batch_imgs).squeeze(0)[1:-1]
        total_outputs.append(outputs)

        outputs = [tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1)) for idx in range(outputs.shape[0])]
        # Save output images
        for idx in range(len(chunk_indices)):
            output_name = names[chunk_indices[idx]]
            output_path = os.path.join(save_path, output_name)
            if output_name not in created_images:
                Image.fromarray(outputs[idx]).save(output_path)
                created_images.add(output_name)
                all_sorted_image_paths.append(output_path)

        print(f"\rProcessing S: {len(created_images)} / {len(images)} ({(len(created_images) / len(images)) * 100:.2f}%)", end="", flush=True)

    total_outputs = torch.cat(total_outputs, dim=0)

    return all_sorted_image_paths, total_outputs
    

def process_ALS(
        model, similarity, images, names, 
        save_path, transform_path,
        thres_values, max_sequence_length, device
    ):
    """Process sequences with given thresholds and save images."""
    created_images = set()
    os.makedirs(save_path, exist_ok=True)

    feature_rankings = np.array(compute_feature_distance_ranking(images))
    position_rankings = np.array(compute_position_distance_ranking(images, transform_path))

    all_images_created = False
    for threshold in thres_values:
        if all_images_created == True:
            break

        reference_indices = list(range(len(names)))
        # random.shuffle(reference_indices)
        # print(f"\nReference indices: {reference_indices}")

        for reference_index in reference_indices:
            if similarity == 'feature':
                sorted_indices = ordering_sim1_thresholding_sim2(
                    images, reference_index, threshold, feature_rankings, position_rankings, similarity
                )
            elif similarity == 'pose':
                sorted_indices = ordering_sim1_thresholding_sim2(
                    images, reference_index, threshold, position_rankings, feature_rankings, similarity
                )
            else:
                raise ValueError("feature or pose only")

            if len(sorted_indices) <= 2 and threshold != thres_values[-1]:
                continue

            if len(sorted_indices) > max_sequence_length:
                sorted_indices = sorted_indices[:max_sequence_length]

            # print(f"Reference index: {reference_index}, Sorted indices: {sorted_indices}")

            all_middle_images_created = True
            middle_indices = list(range(len(sorted_indices)))
            middle_names = [names[sorted_indices[idx]] for idx in middle_indices]
            middle_paths = [os.path.join(save_path, name) for name in middle_names]

            for output_path in middle_paths:
                if not os.path.exists(output_path):
                    all_middle_images_created = False
                    break

            if all_middle_images_created:
                # print(f"Skipping middle image creation for reference index {reference_index} as all middle images already exist.")
                continue
            else:
                batch_imgs = [images[sorted_indices[idx]] for idx in range(len(sorted_indices))]
                batch_imgs = np.stack(batch_imgs).transpose(0, 3, 1, 2)
                batch_imgs = torch.from_numpy(batch_imgs).float().div(255.0).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(batch_imgs).squeeze(0)

                for idx, middle_idx in enumerate(middle_indices):
                    if ((idx == 0) or (idx == len(middle_indices)-1)) and threshold != thres_values[-1]:
                        continue

                    output_img = tensor2img(outputs[middle_idx], rgb2bgr=True, min_max=(0, 1))
                    output_path = middle_paths[idx]
                    output_name = middle_names[idx]

                    if not os.path.exists(output_path):
                        Image.fromarray(output_img).save(output_path)
                        created_images.add(output_name)
                        
                        # print(f"Created: {output_name}")
                
                if len(created_images) == len(names):
                    all_images_created = True
                    
            print(f"\rProcessing ALS: {len(created_images)} / {len(images)} ({(len(created_images) / len(images)) * 100:.2f}%)", end="", flush=True)


def save_sorted_images(image_paths, sorted_indices, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i, idx in enumerate(sorted_indices):
        img = cv2.imread(image_paths[idx])
        cv2.imwrite(os.path.join(output_directory, os.path.basename(image_paths[idx])), img)


def create_video_from_images(image_dirs, video_path, fps=30):
    image_paths = sorted(image_dirs)
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image_path in image_paths:
        video.write(cv2.imread(image_path))
    video.release()

