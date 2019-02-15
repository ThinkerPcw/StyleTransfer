from torchvision import transforms
from PIL import Image
import pylab as plt
import numpy as np


def get_image(path, size):
    m_trans = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    image = Image.open(path)
    image = m_trans(image)
    image = image.unsqueeze(0)
    return image


def show_image(img):
    img = img.squeeze(0)
    pil = transforms.ToPILImage()
    img = pil(img)
    plt.imshow(img)
    plt.show()


def save_image(img, name):
    img = img.squeeze(0)
    pil = transforms.ToPILImage()
    img = np.array(pil(img))
    print(img.shape)
    plt.imsave("result/" + name + ".jpg", img)

