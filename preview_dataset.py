# Angle visualization adapted from: vani-or/car_pose_estimation - https://github.com/vani-or/car_pose_estimation

import matplotlib.pyplot as plt
from dataset import Pascal3DDataset
import math
from PIL import Image, ImageDraw

VEHICLE_VISUALIZATION_IMAGE = "vehicle.jpg"
    
def plot_angle_visualization(angle_pred=None, angle_true=None):
    with Image.open(VEHICLE_VISUALIZATION_IMAGE) as img:
        w, h = img.size

        def draw_direction_line(image, angle, color):
            angle_rad = -angle / 180 * math.pi
            x1 = w // 2
            y1 = h // 2
            x2 = int(x1 + math.cos(angle_rad) * max(w, h))
            y2 = int(y1 + math.sin(angle_rad) * max(w, h))
            image.line([x1, y1, x2, y2], fill=color, width=10)
            return image

        visualization = ImageDraw.Draw(img)

        if angle_pred is not None:
            visualization = draw_direction_line(visualization, angle_pred, "red")

        if angle_true is not None:
            visualization = draw_direction_line(visualization, angle_true, "green")

        return img

def vizualize_image(image_data, y_pred_angle=None, y_true_angle=None, y_pred_bbox=None, y_true_bbox=None):
    image = image_data.numpy().astype("uint8")

    image = Image.fromarray(image, "RGB")
    if y_pred_angle is not None or y_true_angle is not None:
        scheme = plot_angle_visualization(y_pred_angle, y_true_angle)
        scheme.thumbnail(
            (image.size[0] * 0.4, image.size[1] * 0.4), 
            Image.LANCZOS
        )
        image.paste(
            scheme,
            (image.size[0] - scheme.size[0], 0, image.size[0], scheme.size[1]),
        )

    h, w = image.size[:2]
    def draw_bbox(image, bbox, color):
        x_min = int(bbox[0] * w)
        y_min = int(bbox[1] * h)
        x_max = int(bbox[2] * w)
        y_max = int(bbox[3] * h)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x_min, y_min, x_max, y_max], outline=color, width=2
        )
        return image
    
    if y_pred_bbox is not None:
        image = draw_bbox(image, y_pred_bbox, "red")

    if y_true_bbox is not None:
        image = draw_bbox(image, y_true_bbox, "green")

    return image

def show_images():
    dataset = Pascal3DDataset(mode='train', shuffle=False, augment=False)
    ds = dataset.get_dataset(task=None)
    for images, data in ds.take(1):
        batch_size = images.shape[0]
        plt.figure(figsize=(15, 15))
        for i in range(batch_size):
            image = vizualize_image(
                image_data=images[i], 
                y_true_bbox=data["bbox"][i],
                y_true_angle=data["azimuth"][i]
            )
            
            plt.subplot(4, 4, i + 1)
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_images()
