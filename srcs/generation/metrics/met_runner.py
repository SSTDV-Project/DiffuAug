import torch

from DiffuAug.srcs.generation.metrics.fid import compute_fid


def main():
    OPTION = "FID"
    
    if OPTION == "FID":
        DUKE_DATA_PATH = r"/data/duke_data/size_64/split_datalabel"
        FAKE_DATA_PATH = r"/data/results/generation/sampling/cfg/sampling_imgs/ddim/p_uncond_0.2/w_0.0"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # FID 계산
        compute_fid(
            device=device,
            original_data_path=DUKE_DATA_PATH,
            generated_data_path=FAKE_DATA_PATH
            )
        
    elif OPTION == "IS":
        pass

if __name__ == '__main__':
    main()