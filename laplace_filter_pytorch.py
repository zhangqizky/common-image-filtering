import torch.nn as nn
import numpy as np
import torch
import cv2


class Laplacian(nn.Module):
    def __init__(self,kernel_size):
        super(Laplacian,self).__init__()
        self.kernel_size = kernel_size

    def forward(self,x):
        bs = x.size(0)
        res = []
        print(bs)
        for b in range(bs):
            channels = x[b].size(0)
            res_c = []
            for c in range(channels):
                src = x[b][c].numpy()
                src = src*255.
                src[src<0] = 0.
                src[src>255.] = 255.
                src = src.astype('uint8')
                src = cv2.GaussianBlur(src, (self.kernel_size, self.kernel_size), 0)
                dst = cv2.Laplacian(src, cv2.CV_16S, ksize=self.kernel_size)
                abs_dst = cv2.convertScaleAbs(dst)
                cv2.imwrite("test{}.png".format(c),abs_dst)
                res_c.append(abs_dst)
            res.append(res_c)
        res_array = np.array(res)
        return torch.from_numpy(res_array)

if __name__=="__main__":
    img = cv2.imread("000001.png")
    img_array = img.transpose(2, 0, 1).astype(np.float32)
    img_array = img_array/255.
    img_array = torch.from_numpy(img_array)
    # img_array = Variable(img_array.float())
    img_array = img_array.view(-1,img_array.size(-3),img_array.size(-2),img_array.size(-1))
    lap = Laplacian(3)
    out = lap(img_array)
    print(out.shape)

