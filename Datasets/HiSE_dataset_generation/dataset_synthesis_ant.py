import os
import glob

import numpy as np
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import (
    sample_point_in_sphere, random_perturbation, compute_modelview_matrices)
from paz.backend.image import write_image
from paz.pipelines import RandomizeRenderedImage
from utils import color_object, as_mesh
from pyrender import (PerspectiveCamera, OffscreenRenderer, DirectionalLight,
                      RenderFlags, Mesh, Scene)
import trimesh


# ANT packages
import time                             # NOQA
import sys                              # NOQA
sys.path.append('../../..')             # NOQA
from my_functions import print_time     # NOQA


class PixelMaskRenderer():
    def __init__(self, path_OBJ, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self._build_scene(path_OBJ, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.flags_RGBA = RenderFlags.RGBA
        self.flags_FLAT = RenderFlags.RGBA | RenderFlags.FLAT
        self.epsilon = 0.01

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(
            DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(
            PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.pixel_mesh = self.scene.add(color_object(path))
        self.mesh = self.scene.add(
            Mesh.from_trimesh(as_mesh(trimesh.load(path)), smooth=False))  # Note: False should be more realistic, old value - smooth = True
        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.light.light.intensity = intensity
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)
        self.pixel_mesh.mesh.is_visible = False
        image, depth = self.renderer.render(self.scene, self.flags_RGBA)
        self.pixel_mesh.mesh.is_visible = True
        image, alpha = split_alpha_channel(image)
        self.mesh.mesh.is_visible = False
        RGB_mask, _ = self.renderer.render(self.scene, self.flags_FLAT)
        self.mesh.mesh.is_visible = True
        return image, alpha, RGB_mask


if __name__ == "__main__":
    # -------------------------------------------------------------
    # State paths
    # -------------------------------------------------------------
    root_path = ''  # os.environ[AUTOPROJ_CURRENT_ROOT]
    # source_path = './'  # bundles/hise/scripts/dataset_generation/
    source_path = os.path.dirname(os.path.abspath(__file__))
    dataset_name = 'dataset_HiSE256_test'

    obj_path = os.path.join(root_path, source_path, 'models/')
    dataset_path = os.path.join(
        root_path, source_path, dataset_name)  # , 'generated_images/'
    images_path = os.path.join(dataset_path, 'HiSE_images/')
    masks_path = os.path.join(dataset_path, "HiSE_masks/")
    annotation_path = os.path.join(dataset_path, "annotation.txt")
    background_path = os.path.join(
        root_path, source_path, 'background_images/*.png')
    background_list = glob.glob(background_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    print("------------")
    print("Root path: " + root_path)
    print("OBJ path: " + obj_path)
    print("Image path: " + images_path)
    print("Annotation path: " + annotation_path)
    print("Background path: " + background_path)
    print("------------")
    # -------------------------------------------------------------
    # Generic parameters
    # -------------------------------------------------------------
    num_occlusions = 0                  # number of occlusion
    max_radius_scale = 0.5              # ratio of the object occlusion
    # the background images should be bigger than image shape
    # default: (300, 300, 3), (256, 256, 3), (64, 64, 3)
    image_shape = (256, 256, 3)
    viewport_size = image_shape[:2]
    y_fov = np.pi / 4.0
    light = [30, 50]
    top_only = True
    roll = np.pi
    shift = 0.05
    distance = [1, 5]

    num_images = 2                      # number of generated images per label object

    # Labels
    objects = ['blue', 'green', 'red', 'yellow']
    number_classes = len(objects)
    # labels = np.arange(0, number_classes)
    # HiSE256: 5000, HiSE64: 20000, MNIST: 70000, CIFAR10: 60000
    dataset_size = 5000

    # extend_dataset = False
    # if extend_dataset:
    #     open_parameter = 'a'
    #     # proceed_datum = 0               # Where to continue data generation
    # else:
    open_parameter = 'w'
    # proceed_datum = 0

    import shutil
    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)

    dataset = dict()
    dataset['labels'] = dict()
    # dataset_filename = os.path.join(images_path, 'HiSE')

    start_time = time.time()

    with open(annotation_path, open_parameter, encoding="utf-8") as file:  # 'w'/'a'
        for index_datum in range(0, dataset_size):
            object_class = int(np.random.randint(number_classes))
            obj_name = 'cube_' + objects[object_class] + '.obj'
            # print(obj_name)

            renderer = PixelMaskRenderer(obj_path + obj_name, viewport_size, y_fov, distance,
                                         light, top_only, roll, shift)
            randomize_image = RandomizeRenderedImage(background_list, num_occlusions,
                                                     max_radius_scale)

            for image_arg in range(num_images):
                image, alpha, mask = renderer.render()
                RGB_mask = mask[..., 0:3]
                image_filename = f'image_{index_datum:06d}_{image_arg:03d}.png'
                # mask_filename = 'mask_%06d_%03d.png' % (index_datum, image_arg)
                H, W = image.shape[:2]
                # extract the 2D bounding box
                y, x = np.nonzero(image[..., 2])
                x_min = np.min(x)
                y_min = np.min(y)
                x_max = np.max(x)
                y_max = np.max(y)
                # dataset['labels'][image_filename] = [index_datum,
                #                                      object_class, H, W, x_min, y_min, x_max, y_max]
                row = f'{image_filename},{index_datum},{object_class},{H},{W},box,{x_min},{y_min},{x_max},{y_max}'
                file.write(row + "\n")
                image_filename = os.path.join(images_path, image_filename)
                # mask_filename = os.path.join(masks_path, mask_filename)
                image = randomize_image(image, alpha)
                # if index_datum == 0:
                #     dataset['images'] = image[np.newaxis, ...]
                # dataset['images'] = np.concatenate(
                #     [dataset['images'], image[np.newaxis, ...]])
                save_time = time.time()
                # np.savez(dataset_filename, dataset)
                write_image(image_filename, image)
                # TODO: What are the masks for?
                # write_image(mask_filename, RGB_mask)

            if (index_datum + 1) % 100 == 0:
                print(index_datum + 1)
                # print(f"Save time: {time.time() - save_time}")
                print('Total time: ' + print_time(time.time() - start_time))
