from PIL import Image
import numpy as np
import cv2 as cv
import os
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Merge the image slice with mask coloring.")
parser.add_argument("--img_root", default="./Dataset", type=str, help="input image directory")
parser.add_argument("--mask_root", default="./Dataset", type=str, help="input mask directory")
parser.add_argument("--output_root", default="./Dataset", type=str, help="output directory")

if __name__ == "__main__":
    args = parser.parse_args()

    # Process each image file in the specified directory
    for img_file in sorted(os.listdir(args.img_root)):
        if img_file.endswith('.jpg'):
            # Construct full path to the image
            img_path = os.path.join(args.img_root, img_file)
            # Open the grayscale image and convert it to RGB
            # rgb_image = Image.open(img_path).convert('RGB')
            # rgb_image = Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
            # Convert the PIL image to a NumPy array for OpenCV processing
            # rgb_image = np.array(rgb_image)
            # rgb_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
            
            gray_image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            # rgb_image = np.stack([gray_image]*3, axis=0)
            rgb_image=cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
            # image_chw = np.transpose(rgb_image, (2, 0, 1))

            print(f"rgb_image shape: {rgb_image.shape}")

            cv.imwrite(os.path.join(args.output_root, 'tmp.jpg'), rgb_image)
            

            # Check if corresponding mask exists
            mask_file = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(args.mask_root, mask_file)
            if os.path.exists(mask_path):
                # Load the mask
                mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

                # Create a red image of the same size as the RGB image
                redImg = np.zeros_like(rgb_image)
                redImg[:, :] = (0, 0, 255)

                # Apply the mask to the red image
                redMask = cv.bitwise_and(redImg, redImg, mask=mask_img)

                # Blend the red mask with the original RGB image
                cv.addWeighted(redMask, 0.7, rgb_image, 1, 0, rgb_image)

            # Convert back to PIL image to save or display
            result_image = Image.fromarray(cv.cvtColor(rgb_image, cv.COLOR_BGR2RGB))

            # Save or display the resulting image
            result_image.save(os.path.join(args.output_root, f"{img_file}"))
            # plt.imshow(result_image)
            # plt.savefig(os.path.join(args.output_root, f"{img_file}"))

    print(f"Processing complete. Results saved at: {args.output_root}")
