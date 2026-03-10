# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import save
from bokeh.resources import INLINE
from bokeh.plotting import gridplot
from PIL import Image, ImageDraw, ImageFont
import cv2

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png").convert('RGB'), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png").convert('RGB'), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m"
    # mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    
    ############################### MOD ################################
    output_fn = example_dir / "visualizations" / "detections.html"
    output_fn.parent.mkdir(exist_ok=True)
    save(fig_det, filename=output_fn, resources=INLINE, title="detections")

    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:

    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )

    save_predictions(example_dir, output)
    return


def make_output_visualization(
    example_dir: Path,
) -> None:

    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)
    fig_axis = draw_triaxis(rgb, object_datas, camera_data.K) # type: ignore
    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    fig_axis = plotter.plot_image(fig_axis)


    ############################### MOD ################################
    save(fig_mesh_overlay, filename=str(vis_dir / "mesh_overlay.html"), resources=INLINE, title="mesh_overlay")
    save(fig_contour_overlay, filename=str(vis_dir / "contour_overlay.html"), resources=INLINE, title="contour_overlay")
    save(fig_axis, filename=str(vis_dir / "axis.html"), resources=INLINE, title="axis")
    # save(fig_all, filename=str(vis_dir / "all_results.html"), resources=INLINE, title="all_results")

    logger.info(f"Wrote visualizations to {vis_dir}.")
    return


# projection helper
def project_point(p3, K):
    x, y, z = float(p3[0]), float(p3[1]), float(p3[2])
    if z == 0:
        return None
    uv = K @ np.array([x, y, z], dtype=float)
    u = float(uv[0]) / float(uv[2])
    v = float(uv[1]) / float(uv[2])
    return int(round(u)), int(round(v))


def draw_triaxis(oimg, datas, K):
    
    if datas[0] is None:
        return oimg

    T = datas[0].TWO 
    H = T.toHomogeneousMatrix()
    R = H[:3, :3].astype(float)
    t = H[:3, 3].astype(float)

    # Rodrigues rotation vector and translation vector for OpenCV
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Use zero distortion if you don't have distortion coefficients
    dist = np.zeros((5, 1), dtype=float)

    # axis length in same units as your pose translation (adjust as needed)
    axis_length = 0.1

     # Convert RGB -> BGR before OpenCV drawing
    img_bgr = cv2.cvtColor(oimg, cv2.COLOR_RGB2BGR)
    cv2.drawFrameAxes(img_bgr, K, dist, rvec, tvec, axis_length)

    # Convert back BGR -> RGB for display
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)
