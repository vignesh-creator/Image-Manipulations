import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import numpy as np
class preprocess(object):
    def __init__(self, dataset_location=" ", batch_size=1, shuffle=False):
        self.location=dataset_location
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.seed=1
        data=self.__getitem__()
        self.idx=list(data.keys())
        self.imgs=list(data.values())
    def rescale(self, s):
        self.s = s
        self.dic = {}
        for k in range(0, len(self.imgs)):
            image = np.asarray(self.imgs[k], dtype=np.float64)
            img_height, img_width = image.shape[:2]
            height = (self.s) * img_height
            width = (self.s) * img_width
            # Image.fromarray(image).show()
            if len(image.shape) == 2:
                rescale = np.empty([height, width])
                rax = float((img_width - 1) / (width - 1))
                ray = float((img_height - 1) / (height - 1))
                # print(image.size)
                for i in range(0, height):
                    for j in range(0, width):
                        x_min, y_min = math.floor(rax * j), math.floor(ray * i)
                        x_max, y_max = math.ceil(rax * j), math.ceil(ray * i)
                        x_weight = (rax * j) - x_min
                        y_weight = (ray * i) - y_min
                        # print(x_weight,y_weight)
                        pixel = float(image[y_min, x_min] * (1 - x_weight) * (1 - y_weight) + image[y_min, x_max] * x_weight * (1 - y_weight) + image[y_max, x_min] * y_weight * (1 - x_weight) + image[y_max, x_max] * x_weight * y_weight)
                        rescale[i][j] = pixel
                # print(rescale.size)
                # Image.fromarray(rescale).show()
            else:
                rescale = np.empty([height, width,image.shape[2]], np.uint8)
                rax = ((img_width - 1) / (width - 1))
                ray = ((img_height - 1) / (height - 1))
                for i in range(0, height):
                    for j in range(0, width):
                        x_min, y_min = math.floor(rax * j), math.floor(ray * i)
                        x_max, y_max = math.ceil(rax * j), math.ceil(ray * i)
                        x_weight = (rax * j) - x_min
                        y_weight = (ray * i) - y_min
                        pixel = image[y_min, x_min, :] * (1 - x_weight) * (1 - y_weight) + image[y_min, x_max,:] * x_weight * (1 - y_weight) + image[y_max, x_min, :] * y_weight * (1 - x_weight) + image[y_max,x_max,:] * x_weight * y_weight
                        rescale[i, j, :] = pixel
                # Image.fromarray(rescale).show()
            self.dic[self.idx[k]] = rescale
        return self.dic
    def resize(self,h,w):
        self.h_new=h
        self.w_new=w
        self.dic={}
        for k in range(0,len(self.imgs)):
            arr=np.asarray(self.imgs[k],dtype=np.float64)
            height,width=arr.shape[:2]
            h_ratio=self.h_new/(height)
            w_ratio=self.w_new/(width)
            if len(arr.shape)==2:
                req = np.zeros((self.h_new, self.w_new), np.uint8)
                for i in range(self.h_new):
                    for j in range(self.w_new):
                        h_req = int(i / h_ratio)
                        w_req = int(j / w_ratio)
                        req[i, j] = arr[h_req, w_req]
                # Image.fromarray(req).show()
            else:
                req=np.zeros((self.h_new,self.w_new,3),np.uint8)
                for i in range(self.h_new):
                    for j in range(self.w_new):
                        h_req = int(i / h_ratio)
                        w_req = int(j / w_ratio)
                        req[i,j,:] = arr[h_req,w_req,:]
                # Image.fromarray(req).show()
            self.dic[self.idx[k]]=req
        return self.dic
    def crop(self,id1,id2,id3,id4):
        self.id1 = id1
        self.id2 = id2
        self.id3 = id3
        self.id4 = id4
        self.dic = {}
        for k in range(0,len(self.imgs)):
            arr = np.asarray(self.imgs[k],dtype=np.float64)
            if len(arr.shape)==2:
                req=arr[arr.shape[0]-self.id1[1]:arr.shape[0]-self.id4[1],self.id1[0]:self.id2[0]]
                Image.fromarray(req).show()
            else:
                req = np.array(arr[arr.shape[0]-self.id1[1]:arr.shape[0]-self.id4[1], self.id1[0]:self.id2[0],:],np.uint8)
                # Image.fromarray(req).show()
            self.dic[self.idx[k]] = req
        return self.dic
    def blur(self):
        self.dic={}
        for k in range(0,len(self.imgs)):
            arr = np.asarray(self.imgs[k], dtype=np.float64)
            if(len(arr.shape)==2):
                # Image.fromarray(arr).show()
                new_img = np.zeros((arr.shape[0], arr.shape[1]))
                for i in range(0, arr.shape[0]):
                    for j in range(0, arr.shape[1]):
                        if i!=0 and j!=0:
                            new_img[i, j] = np.median(arr[i - 1:i + 2, j - 1:j + 2])
                        elif j==0:
                            new_img[:,j]=arr[:,j]
                        elif i==0:
                            new_img[i,:]=arr[i,:]
                # Image.fromarray(new_img).show()
            else :
                new_img = np.zeros((arr.shape[0], arr.shape[1],arr.shape[2]),np.uint8)
                for t in range(0,20):
                    for i in range(0, arr.shape[0]):
                        for j in range(0, arr.shape[1]):
                            for t in range(arr.shape[2]):
                                if i!=0 and j!=0:
                                    new_img[i, j,t] = np.median(arr[i - 1:i + 2, j - 1:j + 2, t])
                                elif j==0:
                                    new_img[:,j,t]=arr[:,j,t]
                                elif i==0:
                                    new_img[i,:,t]=arr[i,:,t]
                    arr=new_img
                Image.fromarray(new_img).show()
            self.dic[self.idx[k]]=new_img
        return self.dic
    def rgb2gray(self):
        self.dic={}
        for i in range(0,len(self.imgs)):
            arr = np.asarray(self.imgs[i], dtype=np.float64)
            # print(arr)
            if len(arr.shape)==2:
                grey=arr
            else :
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
                # Image.fromarray(grey).show()
            self.dic[self.idx[i]] = grey
        return self.dic
    def __getitem__(self):
        img_list = os.listdir(self.location)
        img_list = sorted(img_list, key=len)
        a = random.sample(img_list, min(len(img_list), self.batch_size))
        ret_dict = {}
        if self.shuffle == True:
            for i in a:
                ret_dict[i] = plt.imread('{}/{}'.format(self.location, i))
        else:
            for i in range(min(len(img_list), self.batch_size)):
                ret_dict[img_list[i]] = plt.imread('{}/{}'.format(self.location, img_list[i]))
        return ret_dict
    def translate(self,tx,ty):
        self.dic = {}
        self.tx=tx
        self.ty=ty
        for i in range(0, len(self.imgs)):
            arr = np.asarray(self.imgs[i], dtype=np.float64)
            if len(arr.shape)==2:
                r,c=arr.shape[0],arr.shape[1]
                flipped=np.flipud(arr)
                req=np.zeros((r,c))
                for p in range(0,r):
                    for q in range(0,c):
                        if p+self.ty>=r or q+self.tx>=c:
                            continue
                        req[p+self.ty,q+self.tx]=flipped[p,q]
                req=np.flipud(req)
                # Image.fromarray(req).show()
            else:
                req = np.empty((arr.shape[0], arr.shape[1],arr.shape[2]),np.uint8)
                r, c = arr.shape[0], arr.shape[1]
                flipped = np.flipud(arr)
                req = np.zeros((r,c,arr.shape[2]),np.uint8)
                for p in range(0, r):
                    for q in range(0, c):
                        for s in range(0,arr.shape[2]):
                            if p + self.ty < r and q + self.tx < c:
                                req[p + self.ty, q + self.tx,s] = flipped[p,q,s]
                req=np.flipud(req)
                # Image.fromarray(req).show()
            self.dic[self.idx[i]]=req
        return self.dic
    def rotate(self, theta):
        self.dic = {}
        for t in range(0, len(self.imgs)):
            arr = np.asarray(self.imgs[t], dtype=np.float64)
            cosine = np.cos(np.radians(theta))
            sine = np.sin(np.radians(theta))
            if len(arr.shape) == 2:
                height = arr.shape[0]
                width = arr.shape[1]
                new_height = round(abs(arr.shape[0] * cosine) + abs(arr.shape[1] * sine)) + 1
                new_width = round(abs(arr.shape[1] * cosine) + abs(arr.shape[0] * sine)) + 1
                output = np.zeros((int(new_height), int(new_width)))
                org_height = round(((arr.shape[0] + 1) / 2) - 1)
                org_width = round(((arr.shape[1] + 1) / 2) - 1)
                new_c_height = round(((new_height + 1) / 2) - 1)
                new_c_width = round(((new_width + 1) / 2) - 1)
                for i in range(height):
                    for j in range(width):
                        y = arr.shape[0] - 1 - i - org_height
                        x = arr.shape[1] - 1 - j - org_width
                        new_y = new_c_height - math.floor(-x * sine + y * cosine)
                        new_x = new_c_width - math.floor(x * cosine + y * sine)
                        if (0 <= new_y < new_height) and (new_y >= 0) and (0 <= new_x < new_width) and (new_x >= 0):
                            output[int(new_y), int(new_x)] = arr[i, j]
                # Image.fromarray(output).show()
            else:
                height = arr.shape[0]
                width = arr.shape[1]
                new_height = round(abs(arr.shape[0] * cosine) + abs(arr.shape[1] * sine)) + 1
                new_width = round(abs(arr.shape[1] * cosine) + abs(arr.shape[0] * sine)) + 1
                output = np.zeros((int(new_height), int(new_width), arr.shape[2]),np.uint8)
                org_height = round(((arr.shape[0] + 1) / 2) - 1)
                org_width = round(((arr.shape[1] + 1) / 2) - 1)
                new_c_height = round(((new_height + 1) / 2) - 1)
                new_c_width = round(((new_width + 1) / 2) - 1)
                for i in range(height):
                    for j in range(width):
                        y = arr.shape[0] - 1 - i - org_height
                        x = arr.shape[1] - 1 - j - org_width
                        new_y = new_c_height - round(-x * sine + y * cosine)
                        new_x = new_c_width - round(x * cosine + y * sine)
                        if (0 <= new_y < new_height) and (new_y >= 0) and (0 <= new_x < new_width) and (new_x >= 0):
                            output[int(new_y), int(new_x), :] = arr[i, j, :]
                # Image.fromarray(output).show()
            self.dic[self.idx[t]]=output
        return self.dic
    def edge_detection(self):
        self.dic = {}
        ker=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        for t in range(0, len(self.imgs)):
            arr = np.asarray(self.imgs[t], dtype=np.float64)
            # print(arr.shape)
            h = arr.shape[0]
            w = arr.shape[1]
            if len(arr.shape)==2:
                req=np.zeros((h+2,w+2))
                req[1:req.shape[0]-1,1:req.shape[1]-1]=arr[:,:]
                Gx=np.zeros((req.shape[0]-2,req.shape[1]-2))
                Gy=np.zeros((req.shape[0]-2,req.shape[1]-2))
                for p in range(0,req.shape[0]-2):
                    for q in range(0,req.shape[1]-2):
                        Gx[p,q]=np.sum(req[p:p+3,q:q+3]*ker)
                        Gy[p,q]=np.sum(req[p:p+3,q:q+3]*(-ker.transpose()))
                req_arr=np.sqrt(np.square(Gx)+np.square(Gy))
                req_arr=req_arr*255.0/req_arr.max()
                # Image.fromarray(req_arr).show()
            else:
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
                h = arr.shape[0]
                w = arr.shape[1]
                req = np.zeros((h + 2, w + 2))
                req[1:req.shape[0] - 1, 1:req.shape[1] - 1] = arr[:, :]
                Gx = np.zeros((req.shape[0] - 2, req.shape[1] - 2))
                Gy = np.zeros((req.shape[0] - 2, req.shape[1] - 2))
                for p in range(0, req.shape[0] - 2):
                    for q in range(0, req.shape[1] - 2):
                        Gx[p, q] = np.sum(req[p:p + 3, q:q + 3] * ker)
                        Gy[p, q] = np.sum(req[p:p + 3, q:q + 3] * -ker.transpose())
                req_arr = np.sqrt(np.square(Gx) + np.square(Gy))
                req_arr = req_arr * 255.0 / req_arr.max()
                # Image.fromarray(req_arr).show()
            self.dic[self.idx[t]]=req_arr
        return self.dic

# inputarg = preprocess("C:/Users/xxx/Desktop/OOP/image_operation/ir_test",21,shuffle =False)
# print(inputarg.rescale(2))
# print(inputarg.resize(200,200))
# print(inputarg.crop((0,90),(200,90),(200,216),(0,216)))
# print(inputarg.rgb2gray())
# print(inputarg.__getitem__())
# print(inputarg.blur())
# print(inputarg.translate(100,100))
# print(inputarg.edge_detection())
# print(inputarg.rotate(45))
