# DIP Assignment – 3-Level Thresholding, CCA, and Mask Evaluation (OpenCV)

This repository contains my Digital Image Processing (DIP) assignment implemented in **Python + OpenCV**.  
The pipeline performs:

1. **Three-level thresholding** (background / uncertain / foreground).
2. **Connected-component labelling (CCA)** to find regions in a target class.
3. **Mask generation & filling** for the largest component.
4. **Inverse/tri-level mask creation** for a second class.
5. **Pixel-wise accuracy** against a provided ground-truth mask.

---

## ✨ What the script does

- **`thresh_3(image)`**  
  Converts a grayscale image into three intensity classes:
  - `0` if `pixel ≤ 65` (background)  
  - `127` if `65 < pixel < 200` (mid/uncertain)  
  - `255` if `pixel ≥ 200` (foreground)

- **`cca_algorithm(n, m, image, V_Set)`**  
  Single-pass CCA over a zero-padded image, labelling only pixels whose value is in `V_Set` (e.g. `{127}` or `{255}`).  
  Uses the **upper 4 neighbors** (`top-left`, `top`, `left`, `top-right`) and merges equivalent labels by assigning the minimum label.

- **`generate_mask(label_matrix)`**  
  Finds the **largest connected component** and fills it using `cv2.findContours` → returns a binary mask (`255` inside the component).

- **`generate_mask_v2(label_matrix_nucleus, threshold_image)`**  
  Produces an **inverse tri-level mask** from the thresholded image:
  - Where threshold is `0` → output `255`
  - Where threshold is `255` → output `0`
  - Else → `127`

- **Accuracy computation**  
  Compares the produced mask with a **ground-truth mask** and prints the **pixel-wise accuracy (coefficient)**.
