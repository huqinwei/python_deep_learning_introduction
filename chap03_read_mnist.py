import numpy as np
from book_dir.dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(img)#np.uint8(img)
    print(type(pil_img))
    pil_img.show()


((train_x,train_y),(test_x,test_y)) = load_mnist()
print(type(train_x),train_x.shape)
print(type(train_y),train_y.shape)#not onehot
print(train_y[:10])
print(test_x.shape)
print(test_y.shape)


((train_x,train_y),(test_x,test_y)) = load_mnist(normalize=False,flatten = True)
img = train_x[0]
label = train_y[0]
print(img.shape)
# img_reshaped = img.reshape(28,28)
img_reshaped = img.reshape(int(np.sqrt(len(img))), int(np.sqrt(len(img))))
print(img_reshaped.shape)
print(img.shape)
img_show(img_reshaped)


