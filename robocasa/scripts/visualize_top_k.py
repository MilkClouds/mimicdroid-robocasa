import os
import cv2
import h5py
import pickle
import tyro
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm

def load_obs_action_slice(
        demo_grp,
        timesteps,              # 1‑D NumPy array or Python list
        obs_keys, action_keys,
    ):
    """
    Return a dict {key: np.ndarray} where every array has shape
    (len(timesteps), feature_dim).  Uses true slicing whenever the
    timesteps form a contiguous block, otherwise fancy‑indexes once.
    """
    ts = np.asarray(timesteps)
    contig = ts[-1] - ts[0] + 1 == len(ts)

    out = {}
    acc_obs = ["obs/"+k if not k.startswith("obs/") else k for k in obs_keys]

    # ---------- observations ----------
    for k, ak in zip(obs_keys, acc_obs):
        ds = demo_grp[ak]
        if 'segmentation_instance' in k:
            # cast it to uint8 for peak memory saving
            out[k] = ds[ts[0]:ts[-1]+1].astype(np.uint8) if contig else ds[ts].astype(np.uint8)
        else:
            out[k] = ds[ts[0]:ts[-1]+1] if contig else ds[ts]

    # ---------- actions ----------
    for k in action_keys:
        ds = demo_grp[k]
        out[k] = ds[ts[0]:ts[-1]+1] if contig else ds[ts]

    return out

def flatten_batch(batch_dict, sep="/"):
    return {k.split(sep)[-1]: v for k, v in batch_dict.items()}

def load_subsequence_cache_hdf5_batch(
        state_paths,
        obs_keys=None,
        action_keys=None,
        flatten=True,
        sep="/",
        use_pbar=False,
):
    """
    state_paths must be sorted by (hdf5_path, demo_key, t).
    Returns `subsequence` – list of per‑timestep dicts in the
    *same* order as `state_paths`, but populated with vectorised
    HDF5 reads.
    """
    i = 0
    subsequence = []
    N = len(state_paths)
    pbar = None
    if use_pbar:
        pbar = tqdm(total=N, desc="Loading subsequence")

    while i < N:
        path, demo, t0 = state_paths[i]
        # ---- find maximal contiguous run belonging to (path, demo) ----
        run_ts = [t0]
        j = i
        while (j + 1 < N
               and state_paths[j + 1][:2] == (path, demo)
               and state_paths[j + 1][2] == state_paths[j][2] + 1):
            j += 1
            run_ts.append(state_paths[j][2])

        with h5py.File(path, 'r', libver='latest') as f:
            root = f.get('data', f)
            grp = root[demo]
            batch = load_obs_action_slice(
                grp, run_ts, obs_keys=obs_keys, action_keys=action_keys,
            )

            if flatten:
                batch = flatten_batch(batch, sep)

            # ------ split the stacked arrays back into per‑step dicts ------
            for step_idx in range(len(run_ts)):
                step = {k: v[step_idx] for k, v in batch.items()}
                subsequence.append(step)

            i = j + 1                       # continue with next run
            if use_pbar:
                pbar.update(len(run_ts))

    return subsequence

def get_index_to_demo_key(dataroot):
    index_to_demo_key_json = os.path.join(dataroot, 'index_to_demo_key.json')
    if not os.path.exists(index_to_demo_key_json):
        raise ValueError(f"index_to_demo_key.json not found in {dataroot}")
    index_to_demo_key = json.load(open(index_to_demo_key_json, 'r'))
    return index_to_demo_key

def generate_hdf5_paths_from_indices(dataroot, indices, hdf5_name=None):
    """
    Generate HDF5 file paths from a list of indices.

    Args:
        dataroot (str): The root directory where index_to_demo_key.json are stored. Typically, this is the path ending with training/validation
        indices (list): A list of indices to generate paths for.

    Returns:
        list: A list of HDF5 file paths.
    """
    index_to_demo_key = get_index_to_demo_key(dataroot)
    hdf5_paths = []
    for i in indices:
        if str(i) not in index_to_demo_key:
            hdf5_paths.append((None, None, None))
            continue
        hdf5_rel_path = index_to_demo_key[str(i)][0]
        if hdf5_name is not None:
            hdf5_rel_path = os.path.join(os.path.dirname(hdf5_rel_path), hdf5_name)
        demo_key = index_to_demo_key[str(i)][1]
        step = index_to_demo_key[str(i)][2]

        hdf5_path = os.path.join(dataroot, hdf5_rel_path)
        if not os.path.exists(hdf5_path):
            print(colored(f"File does not exist: {hdf5_path}", 'red'))
            continue
        hdf5_paths.append((hdf5_path, demo_key, step))
    return hdf5_paths

def get_image_list_from_indices(dataroot, indices, image_key, score=None):
    dataset_paths = generate_hdf5_paths_from_indices(dataroot, indices)
    subsequence = load_subsequence_cache_hdf5_batch(
        state_paths=dataset_paths,
        obs_keys=[image_key],
        action_keys=[],
    )
    images = []
    for s in subsequence:
        img = s[image_key]
        if score is not None:
            # add a text named "score" to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text = f"Score: {score:.2f}"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = img.shape[1] - text_size[0] - 10
            text_y = img.shape[0] - text_size[1] - 10
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        images.append(img[None])
    images = np.concatenate(images, axis=0)
    return images

def visualize_topk_similarity(
    save_vid_dir,
    final_top_k_indices,
    dataroot,
    image_key,
    top_k=10,
    postfix="",
):
    os.makedirs(save_vid_dir, exist_ok=True)
    final_top_k_keys = list(final_top_k_indices.keys())
    random_idx = np.random.choice(final_top_k_keys, 10)

    for idx in random_idx:
        print(f"Visualizing for index {idx}")
        rgb_static_images = []
        index_range = list(range(idx, idx + 256))
        rgb_static = get_image_list_from_indices(
            dataroot,
            index_range,
            image_key=image_key,
        )
        rgb_static_images.append(rgb_static)

        for i in range(top_k):
            entry = final_top_k_indices[idx][i]
            start, end = entry[:2]
            score = entry[2]

            full_indices = list(range(start, end))
            rgb_static = get_image_list_from_indices(
                dataroot,
                full_indices,
                image_key=image_key,
                score=score,
            )
            rgb_static_images.append(rgb_static)

        save_path = os.path.join(save_vid_dir, f"topk_viz_{idx}{postfix}.mp4")
        rgb_static_images = np.concatenate(rgb_static_images, axis=0)
        vid_writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30,
            (rgb_static_images.shape[2], rgb_static_images.shape[1])
        )
        for i in range(rgb_static_images.shape[0]):
            vid_writer.write(cv2.cvtColor(rgb_static_images[i], cv2.COLOR_RGB2BGR))
        vid_writer.release()
    return

def main():
    HOME = os.path.expanduser("~")
    MIMICDROID_ROOT = f"{HOME}/Downloads/MimicDroidDataset"
    save_vid_dir: str = f"{MIMICDROID_ROOT}/training/top-k-vis"
    pkl_file: str = f"{MIMICDROID_ROOT}/training/PlayEnvFinal_dinov2_sl256_top_k_indices.pickle"
    image_key: str = "robot0_agentview_center_image"
    top_k: int = 10
    postfix: str = ""
    # Replace the following with actual loading code as needed
    # final_top_k_indices = torch.load('final_top_k_indices.pt')
    final_top_k_indices = pickle.load(open(pkl_file, 'rb'))
    dataroot = os.path.dirname(pkl_file)
    visualize_topk_similarity(
        save_vid_dir=save_vid_dir,
        final_top_k_indices=final_top_k_indices,
        dataroot=dataroot,
        image_key=image_key,
        top_k=top_k,
        postfix=postfix,
    )

main()
