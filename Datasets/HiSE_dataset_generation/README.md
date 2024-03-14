The scripts given here are for dataset generation using a given set of object models, background images, masks and other variable image parameters. The script that actually generates the images is dataset_synthesis.py. It uses PAZ, which is a perception library in python for image related applications. [PAZ](https://github.com/oarriaga/paz).

## Installation
PAZ has only **three** dependencies: [Tensorflow2.0](https://www.tensorflow.org/), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

Also: pyrender, trimesh!!!

To install PAZ with pypi run:
```
pip install pypaz --user
```
Once PAZ is successfully installed, you can run the script to generate images:

```
python dataset_synthesis.py
```

The generated images will be stored in generated_images/images and the image masks are available in generated_images/masks. The number of images to be generated can be adjusted within the script. 

Have fun! :)

