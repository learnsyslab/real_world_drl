import cv2
import numpy as np
import time

img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

start = time.perf_counter()
for _ in range(1000):
    resized_img = cv2.resize(img, (768, 576), interpolation=cv2.INTER_AREA)

end = time.perf_counter()

print(f"Time taken for 1000 resizes : {end - start:.4f} seconds")

print(f"Resized image shape: {resized_img.shape}")
