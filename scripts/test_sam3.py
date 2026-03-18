import time
import tifffile
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from crisp_drl.envs.sam3_detector import Sam3Detector


def main():
    # Load the model
    t0 = time.time()

    image = Image.open("test_images/demo_img_color.tiff").convert("RGB")
    image = np.array(image)
    t1 = time.time()
    print(f"Image loaded in {t1 - t0:.2f} seconds.")

    detector = Sam3Detector()
    t2 = time.time()
    print(f"Model loaded in {t2 - t1:.2f} seconds.")

    masks_dict = detector.segment_lego(image)
    t3 = time.time()
    print(f"Inference completed in {t3 - t2:.2f} seconds.")

    # Export the masks as pngs
    for name, mask in masks_dict.items():
        Image.fromarray(mask).save(f"test_images/mask_{name}.png")

    print(
        "Masks saved to test_images/mask_lavender.png and test_images/mask_purple.png"
    )
    print("Run test_foundationpose.py to perform pose estimation.")

    plot(image, masks_dict)


# display the results
def plot(image, masks_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot lavender mask
    lavender_mask = masks_dict["lavender"].cpu().numpy()
    h, w = lavender_mask.shape[-2:]
    lavender_mask_reshaped = lavender_mask.reshape(h, w)
    axes[1].imshow(lavender_mask_reshaped, cmap="gray")
    axes[1].set_title("Lavender Brick Mask")
    axes[1].axis("off")

    # Plot purple mask
    purple_mask = masks_dict["purple"].cpu().numpy()
    purple_mask_reshaped = purple_mask.reshape(h, w)
    axes[2].imshow(purple_mask_reshaped, cmap="gray")
    axes[2].set_title("Purple Brick Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
