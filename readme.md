# RWKV-PCSSC: Exploring RWKV Model for Point Cloud Semantic Scene Completion

Semantic Scene Completion (SSC) aims to generate a complete semantic scene from an incomplete input. Existing approaches often employ dense network architectures with a high parameter count, leading to increased model complexity and resource demands. To address these limitations, we propose **RWKV-PCSSC**, a lightweight point cloud semantic scene completion network inspired by the Receptance Weighted Key Value (RWKV) mechanism. Our method features an RWKV Seed Generator (RWKV-SG) for coarse feature aggregation and multiple RWKV Point Deconvolution (RWKV-PD) stages for progressive feature restoration. By leveraging this compact design, RWKV-PCSSC significantly reduces parameter count (4.18×) and improves memory efficiency (1.37×) compared to the state-of-the-art PointSSC, while achieving top performance across indoor (SSC-PC, NYUCAD-PC) and outdoor (PointSSC) datasets, including our newly proposed 3D-FRONT-PC and NYUCAD-PC-V2 datasets.

---

## Installation

Clone the repository and set up the environment:

```bash
# Clone the repository with submodules
git clone --recurse-submodules [https://github.com/shuaiqianende/RWKVPCSSC.git](https://github.com/shuaiqianende/RWKVPCSSC.git)
cd RWKVPCSSC

# Create and activate conda environment
conda create -n rwkvpcssc python=3.9
conda activate rwkvpcssc

# Install main dependencies
pip install -r requirements.txt

# Install PointNet++ ops
cd src/third_party
pip install pointnet2_ops_lib/. 
# Or alternative installation: pip install git+[https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib](https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib)
cd ../..

# Install KNN_CUDA and Ninja
pip install --upgrade [https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl](https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl)
sudo apt-get install ninja-build

# Install ChamferDistancePytorch (CD)
cd src/loss
pip install git+[https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git)
cd ../..
```

---

## Datasets

We provide open-source access to our proposed datasets. Please download them from the following Google Drive links:

- [3D-FRONT-PC](https://drive.google.com/drive/folders/1IJ-6foMeoZVGBipNFOUD9J-PwvzRMHm_?usp=sharing)
- [NYUCAD-PC-V2](https://drive.google.com/drive/folders/1scZQ6mSMDb2uNJXqODQcwFniIiKHait6?usp=sharing)

---

## Training and Testing

We provide scripts to easily train and evaluate the model using default configurations.

Train model on NYUCAD-PC dataset:
python src/train.py experiment=train_nyucad_pc

Train model on SSC-PC dataset:
python src/train.py experiment=train_ssc_pc
