# [DeepDOF: Deep learning extended depth-of-field microscope for fast and slide-free histology](https://www.pnas.org/content/117/52/33051)
Lingbo Jin<sup>1</sup>,  Yubo Tang<sup>1</sup>, Yicheng Wu,  Jackson B. Coole, Melody T. Tan, Xuan Zhao, Hawraa Badaoui, Jacob T. Robinson, Michelle D. Williams, Ann M. Gillenwater, Rebecca R. Richards-Kortum, and Ashok Veeraraghavan

<sup>1</sup> equal contribution

Reference github repository for the paper [Deep learning extended depth-of-field microscope](https://www.pnas.org/content/117/52/33051). Proceedings of the National Academy of Sciences 117.52 (2020) If you use our dataset or code, please cite our paper:

	@article{jin2020deep,
  	  title={Deep learning extended depth-of-field microscope for fast and slide-free histology},
  	  author={Jin, Lingbo and Tang, Yubo and Wu, Yicheng and Coole, Jackson B and Tan, Melody T and Zhao, Xuan and Badaoui, Hawraa and Robinson, Jacob T and Williams, Michelle D and Gillenwater, Ann M and others},
  	  journal={Proceedings of the National Academy of Sciences},
  	  volume={117},
  	  number={52},
  	  pages={33051--33060},
  	  year={2020},
  	  publisher={National Acad Sciences}
	}

## Dataset

Dataset can be downloaded here: [the training, validation, and testing dataset used in the manuscript](https://zenodo.org/record/3922596)

The dataset contains:
- 600 microscopic fluorescence images of proflavine-stained oral cancer resections (10×/0.25-NA, manual refocusing)
- 600 histopathology images of healthy and cancerous tissue of human brain, lungs, mouth, colon, cervix, and breast from The Cancer Genome Atlas (TCGA) Cancer FFPE slides. 
- 600 INRIA Holiday dataset

In total, it contains 1,800 images (each 1,000 × 1,000 pixels; gray scale)

The 1,800 images were randomly assigned to training, validation, and testing sets that contained 1,500; 150; and 150 images, respectively


## Code

### dependencies
Required packages and versions can be found in deepDOF.yml. It can also be used to create a conda environment.


### training
We use a 2 step training process. Step 1 (DeepDOF_step1.py) does not update the optical layer and only trains the U-net. Step 2 (DeepDOF_step2.py) jointly optimizes both the optical layer and the U-net

### testing
To test the trained network with an image, use test_image_all_720um.py


## Reference

Wu, Yicheng, et al. "Phasecam3d—learning phase masks for passive single view depth estimation." 2019 IEEE International Conference on Computational Photography (ICCP). IEEE, 2019.


