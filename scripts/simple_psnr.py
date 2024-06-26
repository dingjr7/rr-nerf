from common import *
import imageio.v3 as imageio
from tqdm import tqdm

name = "lego"
totpsnr = 0
totssim = 0
minpsnr = 1000
maxpsnr = 0

num = 200

with tqdm(total=num) as pbar:
    for i in range(0, num):
        ref = imageio.imread(
            f"../baseline/data/nerf_synthetic/{name}/test/r_{i}.png")
        ref_image = ref.astype(np.float32) / 255.0
        ref_image[..., :3] = srgb_to_linear(ref_image[..., :3])
        ref_image[..., :3] *= ref_image[..., 3:4]
        ref_image[..., :3] = linear_to_srgb(ref_image[..., :3])
        ref_image = ref_image[..., :3]

        out = imageio.imread(f"../nerf_sim/expr/{name}/{i}.png")
        out_image = out.astype(np.float32) / 255.0
        # out_image[..., :3] = srgb_to_linear(out_image[..., :3])
        # out_image[..., :3] *= out_image[..., 3:4]
        # out_image[..., :3] = linear_to_srgb(out_image[..., :3])

        mse = float(compute_error("MSE", out_image, ref_image))
        psnr = mse2psnr(mse)
        ssim = float(compute_error("SSIM", out_image, ref_image))

        totpsnr += psnr
        totssim += ssim
        minpsnr = psnr if psnr < minpsnr else minpsnr
        maxpsnr = psnr if psnr > maxpsnr else maxpsnr
        pbar.update(1)
        pbar.set_description(f'PSNR: {psnr} dB')

print(
    f"PSNR={totpsnr / num} [min={minpsnr} max={maxpsnr}] SSIM={totssim / num}")
