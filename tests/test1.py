import cv2

def edge_detection_and_save(image_path, output_path, threshold1=50, threshold2=150):
    """
    使用Canny边缘检测算法从图像中提取边缘，并将边缘保存为灰度图像。

    参数:
    - image_path: 图像的文件路径。
    - output_path: 保存边缘图像的路径。
    - threshold1: Canny算法的第一个阈值，默认值为50。
    - threshold2: Canny算法的第二个阈值，默认值为150。

    返回:
    - edges: 边缘图像，大小与输入图像相同。
    """
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊来减少噪声
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(blurred_image, threshold1, threshold2)

    # 将边缘图像保存为灰度图
    cv2.imwrite(output_path, edges)

    return edges
# 调用函数进行边缘检测并保存图像
edges_image = edge_detection_and_save('/Users/songyibao/PycharmProjects/MyINR/data/kodak-dataset/kodim05.png', 'edges_output.jpg')

# 如果你还想显示保存后的边缘图像
import matplotlib.pyplot as plt
plt.imshow(edges_image, cmap='gray')
plt.title('Edge Detection Result')
plt.show()
