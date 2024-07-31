# DiffuAug
This repository contains code for training a model to generate breast MRI using Classifier-Free-Diffusion-Guidance and sampling with DDPM and DDIM.

<br/>

## Generation Results
### With tumors
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/25ddfe70-7072-4ed4-9fcd-cbad2932a6b8" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/c24e6916-8021-4a9d-ad08-7d80736f11e7" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/416ac6bf-7607-442c-9f5b-c044a4cf0d0f" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/fdea9e8f-98d4-4751-873a-d614d0a62e74" width="20%" />

### Without tumors
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/e3ee7f81-7405-4ee3-a403-321411265ff2" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/fead596f-61fe-46e7-ad09-82c39a6c1a8b" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/7eebf9c7-d3dd-451a-a9b7-2d888ce80c7c" width="20%" />
<img src="https://github.com/ArtistDeveloper/DiffuAug/assets/40491724/b382736d-0a57-4a22-aac1-3c9aedf0c75a" width="20%" />

<br/>

## Verifying Augmentation Performance through Generative Models
![image](https://github.com/user-attachments/assets/3eb5e2f7-ff4e-4553-85fb-086ad0123e5c)

<br/>

## Setup
### conda
```bash
conda env create -f DiffuAug
conda activate DiffuAug
```

### pip
```bash
pip install -r requirements.txt
```

