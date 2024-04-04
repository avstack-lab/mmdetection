import argparse
import json
import logging
import os

import avapi
import cv2
from tqdm import tqdm

from avstack.environment.objects import Occlusion


def sensor_is_camera(sensor, cam_filter: str = ""):
    sens = sensor.lower()
    return (
        (("cam" in sens) or ("image" in sens))
        and ("depth" not in sens)
        and ("semseg" not in sens)
        and (cam_filter in sens)
    )


def convert_avstack_to_coco(SM, scene_splits, out_file, n_skips=0, cam_filter=""):
    """
    Converts avstack labels to coco format

    Ability to select which cameras to use for traning

    INPUTS:
    SM -- scene manager
    """
    obj_accept = SM.nominal_whitelist_types
    obj_id_map = {o: i for i, o in enumerate(obj_accept)}

    annotations = []
    images = []
    obj_count = 0
    idx_file = 0
    first = True

    # -- loop over scenes in this split
    n_problems = 0
    n_ignored = 0
    for scene in tqdm(scene_splits):
        try:
            SD = SM.get_scene_dataset_by_name(scene)
        except IndexError as e:
            logging.exception(e)
            print(f"Could not process scene {scene}...continuing")
            continue

        # -- loop over sensors
        for agent in SD.sensor_IDs.keys():
            for sensor in SD.sensor_IDs[agent]:
                if not sensor_is_camera(sensor, cam_filter=cam_filter):
                    continue
                print(f"\nagent: {agent}, sensor: {sensor}")
                frames = SD.get_frames(sensor=sensor, agent=agent)
                height, width = None, None
                for idx in tqdm(range(5, len(frames) - 5, n_skips + 1)):
                    frame = frames[idx]

                    # -- image information
                    try:
                        calib = SD.get_calibration(frame, agent=agent, sensor=sensor)
                        objs = SD.get_objects(frame, agent=agent, sensor=sensor)
                        if len(objs) == 0:
                            n_ignored += 1
                            continue
                    except (FileNotFoundError, KeyError) as e:
                        n_problems += 1
                        logging.exception(e)
                        continue
                    if height is None:
                        height = int(calib.height)
                        width = int(calib.width)
                    img_filepath = SD.get_sensor_data_filepath(
                        frame, agent=agent, sensor=sensor
                    )
                    if not os.path.exists(img_filepath):
                        logging.warning(
                            f"Could not find image filepath at {img_filepath}"
                        )
                        n_problems += 1
                        continue
                    else:
                        try:
                            img = cv2.imread(img_filepath)
                            if img is None:
                                raise
                        except Exception as e:
                            n_problems += 1
                            logging.warning(f"Problem reading image at {img_filepath}")
                            continue
                    images.append(
                        dict(
                            id=idx_file,
                            file_name=img_filepath,
                            height=height,
                            width=width,
                        )
                    )
                    idx_file += 1

                    # -- object information
                    for obj in objs:
                        if obj.occlusion == Occlusion.UNKNOWN:
                            raise RuntimeError("Cannot process unknown occlusion")

                        # -- filter based on occlusion
                        if obj.occlusion in [
                            Occlusion.NONE,
                            Occlusion.PARTIAL,
                            Occlusion.MOST,
                            Occlusion.UNKNOWN,
                        ]:
                            bbox_2d = obj.box.project_to_2d_bbox(calib)
                        else:
                            continue

                        # -- box coordinates are measured from top left and are 0-indexed
                        x_min, y_min, x_max, y_max = bbox_2d.box2d
                        x_min = int(x_min)
                        y_min = int(y_min)
                        x_max = int(x_max)
                        y_max = int(y_max)
                        data_anno = dict(
                            image_id=idx_file,
                            id=obj_count,
                            category_id=obj_id_map[obj.obj_type],
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            area=(x_max - x_min) * (y_max - y_min),
                            segmentation=[],
                            iscrowd=0,
                        )
                        annotations.append(data_anno)
                        obj_count += 1

    # -- store annotations
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{"id": i, "name": n} for n, i in obj_id_map.items()],
    )
    json.dump(coco_format_json, open(out_file, "w"))
    print(
        f"{idx_file} valid images; {n_problems} problems; {n_ignored} ignored with this set"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap avstack data to coco format for training"
    )
    parser.add_argument(
        "--dataset",
        choices=["carla", "kitti", "nuscenes", "carla-infrastructure", "carla-joint"],
        help="Choice of dataset",
    )
    parser.add_argument("--subfolder", type=str, help="Save subfolder name")
    parser.add_argument(
        "--data_dir", type=str, help="Path to main dataset storage location"
    )
    parser.add_argument(
        "--n_skips",
        default=0,
        type=int,
        help="Number of skips between frames of a sequence",
    )
    args = parser.parse_args()

    # -- create scene manager and get scene splits
    dataset = args.dataset
    if args.dataset == "carla-joint":
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        cam_filter = ""
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == "carla-infrastructure":
        dataset = "carla"
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        cam_filter = "infra"
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == "carla-joint":
        dataset = "carla"
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        cam_filter = ""
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == "carla-joint":
        raise NotImplemented
    elif args.dataset == "kitti":
        splits_scenes = avapi.kitti.splits_scenes
        raise
    elif args.dataset == "nuscenes":
        SM = avapi.nuscenes.nuScenesManager(args.data_dir, split="v1.0-trainval")
        splits_scenes = avapi.nuscenes.splits_scenes
        cam_filter = ""
    else:
        raise NotImplementedError(args.dataset)

    # -- run main call
    for split in ["train", "val"]:
        print(f"Converting {split}...")
        out_file = f"../data/{dataset}/{args.subfolder}/{split}_annotation_{dataset}_in_coco.json"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        convert_avstack_to_coco(
            SM,
            splits_scenes[split],
            out_file,
            n_skips=args.n_skips,
            cam_filter=cam_filter,
        )
        print(f"done")
