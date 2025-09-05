import numpy as np
import cv2

def generate_image(n, m):
    new_image = np.zeros((n, m), dtype = np.uint8)
    return new_image

def zero_padding(n, m, my_one_image):
    border_image = np.zeros((n + 2, m + 2), dtype=np.uint8)
    border_image[1:n+1, 1:m+1] = my_one_image
    return border_image

def thresh_3(image):
    n, m = image.shape
    new_image = generate_image(n, m)
    for i in range(n):
        for j in range(m):
            if image[i, j] <= 65:
                new_image[i, j] = 0
            elif image[i, j] >= 200:
                new_image[i, j] = 255
            else:
                new_image[i, j] = 127
    return new_image

def cca_algorithm(n, m, image, V_Set):
    my_image = zero_padding(n, m, image)
    label_matrix = np.zeros((n, m), dtype=np.uint16)
    label = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if my_image[i, j] in V_Set:
                neighbors = []
                if i > 1 and j > 1:
                    neighbors.append(label_matrix[i - 2, j - 2])
                if i > 1:
                    neighbors.append(label_matrix[i - 2, j - 1])
                if j > 1:
                    neighbors.append(label_matrix[i - 1, j - 2])
                if i > 1 and j < m:
                    neighbors.append(label_matrix[i - 2, j])

                # Remove zero labels
                neighbor_labels = [lbl for lbl in neighbors if lbl > 0]

                if not neighbor_labels:
                    label_matrix[i - 1, j - 1] = label
                    label += 1
                else:
                    min_label = min(neighbor_labels)
                    label_matrix[i - 1, j - 1] = min_label
                    for x in range(n):
                        for y in range(m):
                            if label_matrix[x, y] in neighbor_labels:
                                label_matrix[x, y] = min_label

    return label_matrix

def generate_mask(label_matrix):
    n, m = label_matrix.shape
    masked_image = np.zeros((n, m), dtype=np.uint8)
    unique_labels, counts = np.unique(label_matrix, return_counts=True)
    unique_labels = unique_labels[1:]
    counts = counts[1:]
    largest_label = unique_labels[np.argmax(counts)]

    for i in range(n):
        for j in range(m):
            if label_matrix[i, j] == largest_label:
                masked_image[i, j] = 255

    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, 255, thickness=cv2.FILLED)

    return masked_image


def generate_mask_v2(label_matrix_nucleus, threshold_image):
    n, m = threshold_image.shape
    output_image2 = np.zeros((n, m), dtype=np.uint8)

    for i in range(n):
        for j in range(m):
            if threshold_image[i, j] == 0:
                output_image2[i, j] = 255
            elif threshold_image[i, j] == 255:
                output_image2[i, j] = 0
            else:
                output_image2[i, j] = 127
    return output_image2
input_image = cv2.imread("C:/Users/dell/Downloads/train/images/004.bmp", 0)
mask_1 = cv2.imread("C:/Users/dell/Downloads/train/masks/004.png", 0)

output_image = thresh_3(input_image)
r, c = output_image.shape
cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
cv2.imshow("Output Image", output_image)

label_matrix = cca_algorithm(r, c, output_image, [127])

label_matrix_2 = cca_algorithm(r, c, output_image, [255])

masked_image = generate_mask(label_matrix)

masked_image_2 = generate_mask_v2(label_matrix_2, output_image)

n, m = masked_image_2.shape
counts = 0
for i in range(n):
    for j in range(m):
        if masked_image_2[i, j] == mask_1[i, j]:
            counts += 1

coeff = float(counts / (np.multiply(n, m)))

print(coeff)

cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
cv2.imshow("Masked Image", masked_image)

cv2.namedWindow("Masked Image for Task 2", cv2.WINDOW_NORMAL)
cv2.imshow("Masked Image for Task 2", masked_image_2)
cv2.waitKey()
