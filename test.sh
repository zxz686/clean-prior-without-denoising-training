#results/My03/180000/CBSD68-224-SPnoisePerChannel-mae/d04 
#-- Average PSNR/SSIM(RGB): 24.97 dB; 0.7044
#-- Average PSNR_Y/SSIM_Y/LPIPS: 26.66/0.7388/0.2394
python main_test_my.py \
        --model_path My03/My03/models/180000_E.pth \
        --name My03/180000/CBSD68-224-SPnoisePerChannel-mae/d04  \
        --opt options/My03/srresnet.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel-mae/d04

python main_test_my.py \
        --model_path My03/My03/models/180000_E.pth \
        --name My03/180000/CBSD68-224-SPnoisePerChannel-mae/d02  \
        --opt options/My03/srresnet.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel-mae/d02

python main_test_my.py \
        --model_path My03/My03/models/180000_E.pth \
        --name My03/180000/CBSD68-224-Gaussian-mae/noisy50  \
        --opt options/My03/srresnet.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-Gaussian-mae/noisy50

python main_test_maeNoise2clean.py \
        --model_path maeNoise2clean/maeNoise2clean/models/75000_E.pth \
        --name maeNoise2clean/75000/CBSD68-224-Gaussian/noisy15  \
        --opt options/maeNoise2clean/maeNoise2clean.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-Gaussian/noisy15  \
        --folder_mae testset/CBSD68-224-Gaussian-mae/noisy15

python main_test_maeNoise2clean.py \
        --model_path maeNoise2clean/maeNoise2clean/models/75000_E.pth \
        --name maeNoise2clean/75000/CBSD68-224-SPnoisePerChannel/d04  \
        --opt options/maeNoise2clean/maeNoise2clean.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel/d04  \
        --folder_mae testset/CBSD68-224-SPnoisePerChannel-mae/d04

python main_test_maeNoise2clean.py \
        --model_path maeNoise2clean/maeNoise2clean/models/125000_E.pth \
        --name maeNoise2clean/75000/CBSD68-224-SPnoisePerChannel/d04  \
        --opt options/maeNoise2clean/maeNoise2clean.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel/d04  \
        --folder_mae testset/CBSD68-224-SPnoisePerChannel-mae/d04
#results/MEM/45000/CBSD68-224-Gaussian/noisy15 
#-- Average PSNR/SSIM(RGB): 33.43 dB; 0.9273
#-- Average PSNR_Y/SSIM_Y/LPIPS: 35.32/0.9413/0.0599
python main_test_mem.py \
        --model_path MEM/MEM/models/45000_E.pth \
        --name MEM/45000/CBSD68-224-Gaussian/noisy15  \
        --opt options/MEM/MEM.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-Gaussian/noisy15  \
        --folder_mae testset/CBSD68-224-Gaussian-mae/noisy15

#results/MEM/55000/CBSD68-224-Gaussian/noisy50 
#-- Average PSNR/SSIM(RGB): 23.98 dB; 0.6243
#-- Average PSNR_Y/SSIM_Y/LPIPS: 25.93/0.6664/0.3226
python main_test_mem.py \
        --model_path MEM/MEM/models/45000_E.pth \
        --name MEM/45000/CBSD68-224-Gaussian/noisy50  \
        --opt options/MEM/MEM.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-Gaussian/noisy50  \
        --folder_mae testset/CBSD68-224-Gaussian-mae/noisy50

python main_test_mem.py \
        --model_path MEM/MEM/models/45000_E.pth \
        --name MEM/45000/CBSD68-224-SPnoisePerChannel/d02  \
        --opt options/MEM/MEM.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel/d02  \
        --folder_mae testset/CBSD68-224-SPnoisePerChannel-mae/d02
#results/MEM/45000/CBSD68-224-SPnoisePerChannel/d04 
#-- Average PSNR/SSIM(RGB): 20.80 dB; 0.4954
#-- Average PSNR_Y/SSIM_Y/LPIPS: 23.03/0.5390/0.5413
python main_test_mem.py \
        --model_path MEM/MEM/models/45000_E.pth \
        --name MEM/45000/CBSD68-224-SPnoisePerChannel/d04  \
        --opt options/MEM/MEM.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel/d04  \
        --folder_mae testset/CBSD68-224-SPnoisePerChannel-mae/d04



python main_test_catt.py \
        --model_path CAtt/CAtt/models/175000_E.pth \
        --name CAtt/175000/CBSD68-224-SPnoisePerChannel/d04  \
        --opt options/CAtt/CAtt.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-SPnoisePerChannel/d04  \
        --folder_mae testset/CBSD68-224-SPnoisePerChannel-mae/d04

python main_test_catt.py \
        --model_path CAtt/CAtt/models/175000_E.pth \
        --name CAtt/175000/CBSD68-224-Speckle/noisy01  \
        --opt options/CAtt/CAtt.json \
        --folder_gt testset/CBSD68-224-Gaussian/original_png  \
        --folder_lq testset/CBSD68-224-Speckle/noisy01  \
        --folder_mae testset/CBSD68-224-Speckle-mae/noisy01