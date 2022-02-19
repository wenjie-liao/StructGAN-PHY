# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:02:16 2021

@author: WenjieLiao
Email: liaowj17@mails.tsinghua.edu.cn / 1243745415@qq.com
    
"""

import cv2, os
import numpy as np
import math
import time


class Pix2Vect():
    def __init__(self,imginput_dir,imggen_dir,design_configs,output_dir):
        #### General parameters
        self.imginput_dir = imginput_dir # input image path
        self.imggen_dir = imggen_dir # generate image path
        self.output_dir = output_dir # the path of extracted data 
        self.img_H = 256 # uniformed image height
        self.img_W = 512 # uniformed image width
        
        #### 建筑轮廓提取所需参数
        self.archioutline_threshold = 0.3 # 判断是否为整体外轮廓的threshold
        self.scale = float(design_configs[-1]) # 图像的像素与墙体的尺寸之间的缩放比例, unit: pixel/m
        self.wall_thick = 0.2 # 假定墙厚0.3m

        #### 剪力墙提取所需参数
        self.min_gap = int(2*(self.img_H/256)) # 用于提取剪力墙的网格线最小间隔
        self.min_walllen = self.min_gap*2 # 最短墙体长度，短于2倍gap个墙体都将被抹去
        self.exbord = self.min_gap*2 # 为保证图像提取准确性，图像边缘进行扩增
        self.uniform_module = 0.05 # 在将像素坐标转化为实际坐标时，指定所有数据都是该模数的倍数，unit:m
        self.ero_kernel = 2 # 进行墙体mask腐蚀时的卷积核尺寸
        self.beam_coupling_threshold_len = 0 # 联肢剪力墙连梁最大跨度，unit: m
        
        #### 梁提取所需参数
        self.max_beam_len = 8 # 梁最大跨度unit: m

        return None


    #### 建筑外轮廓提取
    def Archioutline_ext(self): # 建筑外轮廓提取主函数
        #### 输入建筑图像读取-清洗的前处理
        # 读取图像
        raw_img = cv2.imread(self.imginput_dir)
        # raw_img = self.img_input
        raw_img = cv2.resize(raw_img,(self.img_W,self.img_H))
        # 图像边界扩展
        rgb_white = [255,255,255]
        bord_img = cv2.copyMakeBorder(raw_img,self.exbord,self.exbord,self.exbord,self.exbord, 
                                      cv2.BORDER_CONSTANT,value=rgb_white)
        # 图像二值化
        gary_img = cv2.cvtColor(bord_img,cv2.COLOR_BGR2GRAY)
        ret,bin_img = cv2.threshold(gary_img, 220, 255, cv2.THRESH_BINARY_INV)
        # mask图像膨胀，让可能存在缝隙的外轮廓全部填满
        kernel = np.ones((3,3),np.uint8)
        dilbin_img = cv2.dilate(bin_img,kernel,iterations = 4)
        # 保存bin图像
        self.binimginput_save_dir = self.imginput_dir.split(".png")[0]+"_bin.png"
        cv2.imwrite(self.binimginput_save_dir,dilbin_img)
        # 轮廓提取
        contours, hierarchy = cv2.findContours(dilbin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #### 计算外轮廓属性
        outline_cnt_pix,outline_cenpts_pix,cnt_props,max_area = [],[],[],0
        for i,contour in enumerate(contours):
            cnt_M = cv2.moments(contour) # 轮廓求矩
            cnt_area = cnt_M['m00'] # 轮廓面积，也可用cv2.contourArea()计算
            x_cm = int(cnt_M['m10']/cnt_M['m00']) # 轮廓中心x
            y_cm = int(cnt_M['m01']/cnt_M['m00']) # 轮廓中心y
            cnt_props.append([contour,cnt_M,cnt_area,(x_cm,y_cm)]) # 储存矩\面积\中心点元组
            if max_area < cnt_area:
                max_area = cnt_area
            else:
                continue
            if max_area >= (self.archioutline_threshold*self.img_W*self.img_H) and len(contour)>4: #如果该轮廓的面积大于整张图像面积*阈值，并且非4节点
                outline_cnt_pix = contour # 整体的外轮廓
                outline_cenpts_pix = (x_cm,y_cm) # 整体外轮廓的中心
                story_pixarea = max_area # 楼层面积
        #### 对建筑外轮廓提取结果进行判断
        if len(outline_cnt_pix) != 0: # 如果有合理的外轮廓
            outline_cnt_img = cv2.drawContours(bord_img,[outline_cnt_pix],-1,(255,0,0),2)
            outline_cnt_img = cv2.circle(outline_cnt_img, outline_cenpts_pix, 5, (255,0,0),-1)
            # 保存图像
            self.imginput_save_dir = self.imginput_dir.split(".png")[0]+"_outline.png"
            cv2.imwrite(self.imginput_save_dir,outline_cnt_img)
        else: # 如果没有合理的外轮廓，则取最大面积轮廓的外接矩形作为可行解
            cnt_props.sort(key=lambda x: x[2], reverse=True) # 根据列表中元组的第3个元素（整体轮廓面积）进行排序
            max_cnt = cnt_props[0][0]
            cntbound_x,cntbound_y,cntbound_w,cntbound_h = cv2.boundingRect(max_cnt)
            outline_cnt_pix = np.array([[cntbound_x,cntbound_y],
                                    [(cntbound_x+cntbound_w),cntbound_y],
                                    [(cntbound_x+cntbound_w),(cntbound_y+cntbound_h)],
                                    [(cntbound_x),(cntbound_y+cntbound_h)]])
            outline_cnt_pix = np.expand_dims(outline_cnt_pix, 1) #增加一个维度
            outline_cnt_img = cv2.drawContours(bord_img,[outline_cnt_pix],-1,(255,0,0),2) # 当无法求出合适的轮廓时，采用最大轮廓外接矩形作为轮廓边界
            cnt_M = cv2.moments(outline_cnt_pix) # 轮廓求矩
            cnt_area = cnt_M['m00'] # 轮廓面积，也可用cv2.contourArea()计算
            x_cm = int(cnt_M['m10']/cnt_M['m00']) # 轮廓中心x
            y_cm = int(cnt_M['m01']/cnt_M['m00']) # 轮廓中心y
            outline_cenpts_pix = (x_cm,y_cm) # 整体外轮廓的中心
            story_pixarea = cnt_area # 楼层面积
            outline_cnt_img = cv2.drawContours(bord_img,[outline_cnt_pix],-1,(255,0,0),2)
            outline_cnt_img = cv2.circle(outline_cnt_img, outline_cenpts_pix, 5, (255,0,0),-1)
            # 保存图像
            self.imginput_save_dir = self.imginput_dir.split(".png")[0]+"_outline.png"
            cv2.imwrite(self.imginput_save_dir,outline_cnt_img)
            
        self.outline_cnt_pix,self.outline_cenpts_pix,self.story_pixarea = outline_cnt_pix,outline_cenpts_pix,story_pixarea
        self.story_area = self.story_pixarea*self.scale**2
            
        #### 近似获取建筑长宽范围
        outline_cntbox_pix = cv2.boundingRect(outline_cnt_pix)
        self.outline_cnt_width= outline_cntbox_pix[2]*self.scale
        self.outline_cnt_height = outline_cntbox_pix[3]*self.scale
        
        return None


    #### 剪力墙图像初步提取
    def shearwall_pro_image(self):
        # 读取图像
        raw_gen_img = cv2.imread(self.imggen_dir)
        # raw_gen_img = self.img_gen
        raw_gen_img = cv2.resize(raw_gen_img,(self.img_W,self.img_H))
        # 图像边界扩展
        rgb_white = [255,255,255]
        bord_img = cv2.copyMakeBorder(raw_gen_img,self.exbord,self.exbord,self.exbord,self.exbord, 
                                      cv2.BORDER_CONSTANT,value=rgb_white)
        # 对图像去除噪音
        blur_img = cv2.bilateralFilter(bord_img,9,75,75)
        # 转换到HSV
        hsv_img = cv2.cvtColor(blur_img,cv2.COLOR_BGR2HSV)
        # 设定shear wall的阈值
        lower_red1, lower_red2 = np.array([0,50,50]),np.array([160,50,50])
        upper_red1, upper_red2 = np.array([10,255,255]),np.array([180,255,255])
        # 根据阈值构建掩模
        mask1, mask2 = cv2.inRange(hsv_img,lower_red1,upper_red1),cv2.inRange(hsv_img,lower_red2,upper_red2)
        mask = mask1 + mask2
        # 对原图像和掩模进行位运算
        res_img = cv2.bitwise_and(blur_img,blur_img,mask=mask)
        gary_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)
        ret,bin_img = cv2.threshold(gary_img,50,255,0)
        # 先腐蚀再膨胀去除噪音
        kernel = np.ones((self.ero_kernel,self.ero_kernel),np.uint8)
        erobin_img = cv2.erode(bin_img,kernel,iterations = 1)
        dilbin_img = cv2.dilate(erobin_img,kernel,iterations = 2)
        # 存储处理后的binary图像
        self.binimggen_save_dir = self.imggen_dir.split(".png")[0]+"_bin.png"
        cv2.imwrite(self.binimggen_save_dir,dilbin_img)
        self.dilbin_img = dilbin_img
        
        return dilbin_img,bord_img

    #### 创建网格线提取剪力墙
    def shearwall_grid_make(self,dilbin_img):
        hori_gridlines,vert_gridlines = [],[]
        dilbin_img_H, dilbin_img_W = dilbin_img.shape[0],dilbin_img.shape[1]
        hori_gridlines_num = math.ceil(dilbin_img_H/self.min_gap)
        vert_gridlines_num = math.ceil(dilbin_img_W/self.min_gap)
        for hori_num in range(hori_gridlines_num):
            hori_gridline = [0,hori_num*self.min_gap,dilbin_img_W,hori_num*self.min_gap]
            hori_gridlines.append(hori_gridline)
        for vert_num in range(vert_gridlines_num):
            vert_gridline = [vert_num*self.min_gap,0,vert_num*self.min_gap,dilbin_img_H]
            vert_gridlines.append(vert_gridline)
        
        self.hori_gridlines = hori_gridlines
        self.vert_gridlines = vert_gridlines
        
        return hori_gridlines,vert_gridlines

    #### 网格与墙体交线求解
    def shearwall_grid_wall_inter(self,dilbin_img):
        dilbin_img_H, dilbin_img_W = dilbin_img.shape[0],dilbin_img.shape[1]
        # 求初始的水平方向墙体
        hori_init_walls,hori_init_wall_lens = [],[]
        for i,hori_gridline in enumerate(self.hori_gridlines):
            hori_inters,hori_inter_lens = [],[]
            hori_gridline_y = int(hori_gridline[1])
            start_x, end_x = 0, 0 # 初始化起点与重点坐标值
            for inter_pt in range(dilbin_img_W): # 通过判断grid与墙体的交线情况，来提取剪力墙坐标
                if dilbin_img[hori_gridline_y,inter_pt] == 255 and start_x == 0:
                    start_x = inter_pt
                if dilbin_img[hori_gridline_y,inter_pt] == 0 and start_x != 0:
                    end_x = inter_pt
                    hori_inter_len = (end_x-start_x) # 当前交线长度
                    if hori_inter_len > self.min_walllen: # 交线长度大于最小墙体长度，认为是墙体
                        hori_inters.append([start_x,hori_gridline_y,end_x,hori_gridline_y]) # 保存墙体坐标
                        hori_inter_lens.append(hori_inter_len) # 保存墙体长度
                    start_x, end_x = 0, 0 # 重置起点与重点坐标值
            if len(hori_inters) != 0: # 如果存在交线墙体，则保存当前水平grid的交线集
                hori_init_walls.append(hori_inters)
                hori_init_wall_lens.append(hori_inter_lens)
        # 求初始的竖直方向墙体
        vert_init_walls,vert_init_wall_lens = [],[]
        for j,vert_gridline in enumerate(self.vert_gridlines):
            vert_inters,vert_inter_lens = [],[]
            vert_gridline_x = int(vert_gridline[0])
            start_y, end_y = 0, 0 # 初始化起点与重点坐标值
            for inter_pt in range(dilbin_img_H): # 通过判断grid与墙体的交线情况，来提取剪力墙坐标
                if dilbin_img[inter_pt,vert_gridline_x] == 255 and start_y == 0:
                    start_y = inter_pt
                if dilbin_img[inter_pt,vert_gridline_x] == 0 and start_y != 0:
                    end_y = inter_pt
                    vert_inter_len = (end_y-start_y) # 当前交线长度
                    if vert_inter_len > self.min_walllen: # 交线长度大于最小墙体长度，认为是墙体
                        vert_inters.append([vert_gridline_x,start_y,vert_gridline_x,end_y])
                        vert_inter_lens.append(vert_inter_len)
                    start_y, end_y = 0, 0 # 重置起点与重点坐标值
            if len(vert_inters) != 0: # 如果存在交线墙体，则保存当前竖直grid的交线集
                vert_init_walls.append(vert_inters)
                vert_init_wall_lens.append(vert_inter_lens)
        # 存储初始墙线组
        self.hori_init_walls, self.vert_init_walls = hori_init_walls,vert_init_walls
        self.hori_init_wall_lens, self.vert_init_wall_lens = hori_init_wall_lens,vert_init_wall_lens
        
        return hori_init_walls,vert_init_walls

    #### 初始水平墙线的清洗
    def shearwall_grid_wall_wash(self):
        num_hori_init_walls, num_vert_init_walls = len(self.hori_init_walls),len(self.vert_init_walls)
        hori_walls,vert_walls,hori_wall_lens,vert_wall_lens = [],[],[],[]
        # 水平墙体清洗
        for i,hori_init_wall in enumerate(self.hori_init_walls):
            # 逐剪力墙比较
            for j,hori_wall_j in enumerate(hori_init_wall):
                j_x1,j_x2 = hori_wall_j[0],hori_wall_j[2]
                if i < (num_hori_init_walls-2): # 后面还有两组墙集
                    for post_num in range(2):
                        post_hori_init_wall = self.hori_init_walls[i+1+post_num] # 后1+post_num组交线集
                        if len(post_hori_init_wall) != 0: # 如果后续数组非0
                            gap_hori_lines = abs(post_hori_init_wall[0][1] - hori_init_wall[0][1]) # 相邻两组交线集的gap
                            if gap_hori_lines <= 2*self.min_gap: # 两组交线集隔的近，表明有交叉重叠部分
                                del_wall_ks = []
                                for k,hori_wall_k in enumerate(post_hori_init_wall):
                                    k_x1,k_x2 = hori_wall_k[0],hori_wall_k[2]
                                    if (0.5*(k_x1 + k_x2) >= (j_x1 - self.min_gap) and  0.5*(k_x1 + k_x2) <= (j_x2 + self.min_gap)) \
                                        or (0.5*(j_x1 + j_x2) >= (k_x1 - self.min_gap) and  0.5*(j_x1 + j_x2) <= (k_x2 + self.min_gap)): # 两个墙体重叠
                                        j_x1 = min(j_x1,k_x1) # 两墙重叠，取两墙包络，并删除后者
                                        j_x2 = max(j_x2,k_x2)
                                        del_wall_ks.append(k)
                                del_wall_ks.reverse() # 倒叙列表
                                for del_wall_k in del_wall_ks:
                                    self.hori_init_walls[i+1+post_num].pop(del_wall_k) # 根据记录删除墙体
                                self.hori_init_walls[i][j][0],self.hori_init_walls[i][j][2] = j_x1,j_x2
                elif i == (num_hori_init_walls-2) : # 后面还有1组墙集
                    post_hori_init_wall = self.hori_init_walls[i+1] # 后1组交线集
                    if len(post_hori_init_wall) != 0: # 如果后续数组非0
                        gap_hori_lines = abs(post_hori_init_wall[0][1] - hori_init_wall[0][1]) # 相邻两组交线集的gap
                        if gap_hori_lines <= 2*self.min_gap: # 两组交线集隔的近，表明有交叉重叠部分
                            del_wall_ks = []
                            for k,hori_wall_k in enumerate(post_hori_init_wall):
                                k_x1,k_x2 = hori_wall_k[0],hori_wall_k[2]
                                if (0.5*(k_x1 + k_x2) >= (j_x1 - self.min_gap) and  0.5*(k_x1 + k_x2) <= (j_x2 + self.min_gap)) \
                                    or (0.5*(j_x1 + j_x2) >= (k_x1 - self.min_gap) and  0.5*(j_x1 + j_x2) <= (k_x2 + self.min_gap)): # 两个墙体重叠
                                    j_x1 = min(j_x1,k_x1) # 两墙重叠，取两墙包络，并删除后者
                                    j_x2 = max(j_x2,k_x2)
                                    del_wall_ks.append(k)
                            del_wall_ks.reverse() # 倒叙列表
                            for del_wall_k in del_wall_ks:
                                self.hori_init_walls[i+1].pop(del_wall_k) # 根据记录删除墙体
                            self.hori_init_walls[i][j][0],self.hori_init_walls[i][j][2] = j_x1,j_x2
                else: # 最后一组
                    self.hori_init_walls[i][j]
                # 对比结束后将更新的墙坐标存储
                if len(self.hori_init_walls[i][j]) != 0:
                    hori_walls.append(self.hori_init_walls[i][j])
                    hori_wall_len = abs(self.hori_init_walls[i][j][2]-self.hori_init_walls[i][j][0])
                    hori_wall_lens.append(hori_wall_len)
        # 竖直墙体清洗
        for i,vert_init_wall in enumerate(self.vert_init_walls):
            # 逐剪力墙比较
            for j,vert_wall_j in enumerate(vert_init_wall):
                j_y1,j_y2 = vert_wall_j[1],vert_wall_j[3]
                if i < (num_vert_init_walls-2): # 后面还有两组墙集
                    for post_num in range(2):
                        post_vert_init_wall = self.vert_init_walls[i+1+post_num] # 后1+post_num组交线集
                        if len(post_vert_init_wall) != 0: # 如果后续数组非0
                            gap_vert_lines = abs(post_vert_init_wall[0][0] - vert_init_wall[0][0]) # 相邻两组交线集的gap
                            if gap_vert_lines <= 2*self.min_gap: # 两组交线集隔的近，表明有交叉重叠部分
                                del_wall_ks = []
                                for k,vert_wall_k in enumerate(post_vert_init_wall):
                                    k_y1,k_y2 = vert_wall_k[1],vert_wall_k[3]
                                    if (0.5*(k_y1 + k_y2) >= (j_y1 - self.min_gap) and  0.5*(k_y1 + k_y2) <= (j_y2 + self.min_gap)) \
                                        or (0.5*(j_y1 + j_y2) >= (k_y1 - self.min_gap) and  0.5*(j_y1 + j_y2) <= (k_y2 + self.min_gap)): # 两个墙体重叠
                                        j_y1 = min(j_y1,k_y1) # 两墙重叠，取两墙包络，并删除后者
                                        j_y2 = max(j_y2,k_y2)
                                        del_wall_ks.append(k)
                                del_wall_ks.reverse() # 倒叙列表
                                for del_wall_k in del_wall_ks:
                                    self.vert_init_walls[i+1+post_num].pop(del_wall_k) # 根据记录删除墙体
                                self.vert_init_walls[i][j][1],self.vert_init_walls[i][j][3] = j_y1,j_y2
                elif i == (num_vert_init_walls-2) : # 后面还有1组墙集
                    post_vert_init_wall = self.vert_init_walls[i+1] # 后1组交线集
                    if len(post_vert_init_wall) != 0: # 如果后续数组非0
                        gap_vert_lines = abs(post_vert_init_wall[0][0] - vert_init_wall[0][0]) # 相邻两组交线集的gap
                        if gap_vert_lines <= 2*self.min_gap: # 两组交线集隔的近，表明有交叉重叠部分
                            del_wall_ks = []
                            for k,vert_wall_k in enumerate(post_vert_init_wall):
                                k_y1,k_y2 = vert_wall_k[1],vert_wall_k[3]
                                if (0.5*(k_y1 + k_y2) >= (j_y1 - self.min_gap) and  0.5*(k_y1 + k_y2) <= (j_y2 + self.min_gap)) \
                                    or (0.5*(j_y1 + j_y2) >= (k_y1 - self.min_gap) and  0.5*(j_y1 + j_y2) <= (k_y2 + self.min_gap)): # 两个墙体重叠
                                    j_y1 = min(j_y1,k_y1) # 两墙重叠，取两墙包络，并删除后者
                                    j_y2 = max(j_y2,k_y2)
                                    del_wall_ks.append(k)
                            del_wall_ks.reverse() # 倒叙列表
                            for del_wall_k in del_wall_ks:
                                self.vert_init_walls[i+1].pop(del_wall_k) # 根据记录，从后往前删除墙体
                            self.vert_init_walls[i][j][1],self.vert_init_walls[i][j][3] = j_y1,j_y2
                else:
                    self.vert_init_walls[i][j]
                # 对比结束后将更新的墙坐标存储
                if len(self.vert_init_walls[i][j]) != 0:
                    vert_walls.append(self.vert_init_walls[i][j])
                    vert_wall_len = abs(self.vert_init_walls[i][j][3]-self.vert_init_walls[i][j][1])
                    vert_wall_lens.append(vert_wall_len)
        # 存储水平与竖直墙体组
        self.hori_walls,self.vert_walls = hori_walls, vert_walls
        self.hori_wall_lens,self.vert_wall_lens = hori_wall_lens,vert_wall_lens
        
        return hori_walls, vert_walls, hori_wall_lens, vert_wall_lens


    #### 根据真实坐标计算墙体长度
    def shearwall_walllens(self,walllines,walldir):
        walllens = []
        for wallline in walllines:
            if walldir == "x":
                walllen = abs(wallline[2] - wallline[0])
            else:
                walllen = abs(wallline[3] - wallline[1])
            walllens.append(walllen)
        
        return walllens

    #### 像素坐标到真实坐标转化
    def coords_pix2real(self,coords_pix):
        coords_real = []
        for coord_pix in coords_pix:
            coords = []
            for coord in coord_pix:
                coord = math.floor(coord*self.scale/self.uniform_module)*self.uniform_module
                coords.append(coord)
            coords_real.append(coords)
            
        return coords_real


    #### 剪力墙提取主函数
    def Shearwall_ext(self):
        #### 更新剪力墙提取的参数
        self.min_gap = int(max(math.floor(self.wall_thick/self.scale),2)*(self.img_H/256)) # 用于提取剪力墙的网格线最小间隔
        self.min_walllen = self.min_gap*3 # 最短墙体长度，短于3倍gap个墙体都将被抹去

        #### 根据gridlines与墙体的交线进行墙线搜索-清洗-输出
        # Attention! 此处可以根据具体问题替换，本研究通过网格线与轮廓线的交点提取墙线
        # 生成剪力墙图像读取-清洗的前处理
        dilbin_img,bord_img = self.shearwall_pro_image()
        # 创建网格线
        hori_gridlines,vert_gridlines = self.shearwall_grid_make(dilbin_img)
        # 网格线与墙线相交求初始墙线坐标
        hori_init_walls,vert_init_walls = self.shearwall_grid_wall_inter(dilbin_img)
        # 初始墙线清洗
        x_walllines_pix,y_walllines_pix,x_walllens_pix,y_walllens_pix = self.shearwall_grid_wall_wash()

        #### 完成所有墙线的提取和清洗
        walllines_pix = x_walllines_pix + y_walllines_pix
        walls_img = np.ones(bord_img.shape,np.uint8)*255
        walls_img[:,:,0] = dilbin_img
        walls_img[:,:,1] = dilbin_img
        walls_img[:,:,2] = dilbin_img
        for wallline in walllines_pix:
            cv2.line(walls_img,(wallline[0],wallline[1]),
                     (wallline[2],wallline[3]),(0,0,255),2)
        self.imggen_save_dir = self.imggen_dir.split(".png")[0]+"_wall.png"
        cv2.imwrite(self.imggen_save_dir,walls_img)

        self.walllines_pix,self.x_walllines_pix,self.y_walllines_pix = walllines_pix,x_walllines_pix,y_walllines_pix
        self.x_walllines = self.coords_pix2real(self.x_walllines_pix)
        self.y_walllines = self.coords_pix2real(self.y_walllines_pix)
        self.walllines = self.x_walllines + self.y_walllines
        self.outline_cnt = self.coords_pix2real(np.squeeze(self.outline_cnt_pix))
        self.outline_cenpts = self.coords_pix2real([self.outline_cenpts_pix])[0]
        self.x_walllens,self.y_walllens = self.shearwall_walllens(self.x_walllines,"x"),self.shearwall_walllens(self.y_walllines,"y")
        self.walllens = self.x_walllens + self.y_walllens

        return None
    
    
    #### output all the extracted data
    def Elements_output(self):
        # output shear wall coordinates as txt file
        self.shearwall_save_dir = self.output_dir+"_shear_wall.txt"
        # format (x1, y1, x2, y2)
        np.savetxt(self.shearwall_save_dir,np.array(self.walllines),fmt='%.05f')
        
        # output story_area, unit: m2
        self.story_area_save_dir = self.output_dir+"_story_area.txt"
        np.savetxt(self.story_area_save_dir,np.array([self.story_area]),fmt='%.05f')
        
        # output architecture floor outline coordinates as txt file
        self.floor_outline_save_dir = self.output_dir+"_floor_outline.txt"
        # format (x_i, y_i)
        np.savetxt(self.floor_outline_save_dir,np.array(self.outline_cnt),fmt='%.05f')
        
        
        return None


#### main function
if __name__ == '__main__':
    raw_imggen_root = ".\\test_B"
    raw_imggen_names = os.listdir(raw_imggen_root)
    raw_imginput_root = ".\\test_A"
    raw_imginput_names = os.listdir(raw_imginput_root)
    raw_design_config_root = ".\\test"
    output_dir_root = ".\\test_output"
    if not os.path.exists(output_dir_root):
        os.makedirs(output_dir_root)
    for i,raw_imggen_name in enumerate(raw_imggen_names):
        for j,raw_imginput_name in enumerate(raw_imginput_names):
            if (raw_imggen_name[-9:]=="image.png") and (raw_imginput_name[-9:]=="label.png") and (raw_imggen_name[:8] == raw_imginput_name[:8]):
                start_time = time.time()
                # find data path
                raw_imgname = raw_imginput_name.split("_input_label.png")[0]
                raw_imggen_path = os.path.join(raw_imggen_root, raw_imggen_name)
                raw_imginput_path = os.path.join(raw_imginput_root, raw_imginput_name)
                raw_design_config_path = os.path.join(raw_design_config_root,raw_imgname+".txt")
                output_dir = os.path.join(output_dir_root,raw_imgname)
                
                # read design config
                with open(raw_design_config_path) as design_config_txt:
                    design_configs = design_config_txt.readline()
                    design_configs = design_configs.split(",")
                    
                # class pix2vect
                imginput_dir,imggen_dir = raw_imginput_path,raw_imggen_path
                pix2vect = Pix2Vect(imginput_dir,imggen_dir,design_configs,output_dir)
                
                # architecture image outline
                print("1: Archioutline_extracted")
                pix2vect.Archioutline_ext()
                
                # structural image shear wall 
                print("2: Shearwall_extracted")
                pix2vect.Shearwall_ext()
                
                # output vector files
                print("3: Vector_elements_extracted")
                pix2vect.Elements_output()

                # print time
                end_time = time.time()
                cost_time = end_time - start_time
                print("time cost: %f \n" %cost_time)