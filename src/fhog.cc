/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, University of Nizhny Novgorod, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


//Modified from latentsvm module's "lsvmc_featurepyramid.cpp".

//#include "precomp.hpp"
//#include "_lsvmc_latentsvm.h"
//#include "_lsvmc_resizeimg.h"

#include "fhog.hpp"


#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif



/**********************************************************************************
函数功能：计算image的hog特征，结果在map结构中的map大小为sizeX*sizeY*NUM_SECTOR*3（Getting feature map for the selected subimage）
函数参数:选中的子图，cell的大小，返回的特征图
RESULT：Error status
// Getting feature map for the selected subimage
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
//总体过程是：
//1.计算每个像素的水平梯度dx和垂直梯度dy
//2.计算每个像素的通道间最大梯度大小r及其最邻近梯度方向的索引值
//3.计算每个block(2+4+2)*(2+4+2)的梯度直方图（分为9和18bin）存于map中//每个block的特征是以一个cell为中心，根据像素的位置以及像素点的梯度强度进行加权获得的
***********************************************************************************/
int getFeatureMaps(const IplImage* image, const int k, CvLSVMFeatureMapCaskade **map)
{
    int sizeX, sizeY;
    int p, px, stringSize;
    int height, width, numChannels;
    int i, j, kk, c, ii, jj, d;
    float  * datadx, * datady;
    
    int   ch; 
    float magnitude, x, y, tx, ty;
    
    IplImage * dx, * dy;
    int *nearest;
    float *w, a_x, b_x;
	// 横向和纵向的3长度{-1，0，1}矩阵 
    float kernel[3] = {-1.f, 0.f, 1.f};
    CvMat kernel_dx = cvMat(1, 3, CV_32F, kernel);// 1*3的矩阵 
	CvMat kernel_dy = cvMat(3, 1, CV_32F, kernel);  // 3*1的矩阵  

    float * r;//记录每个像素点的每个通道的最大梯度
    int   * alfa;//记录每个像素的梯度方向的索引值，分别为9份时的索引值和18份时的索引值。
    
    float boundary_x[NUM_SECTOR + 1];
    float boundary_y[NUM_SECTOR + 1];
    float max, dotProd;
    int   maxi;

    height = image->height;
    width  = image->width ;

    numChannels = image->nChannels;
	// 采样图像大小的Ipl图像 
    dx    = cvCreateImage(cvSize(image->width, image->height), 
                          IPL_DEPTH_32F, 3);
    dy    = cvCreateImage(cvSize(image->width, image->height), 
                          IPL_DEPTH_32F, 3);
	// 向下取整的（边界大小/4），k = cell_size
    sizeX = width  / k;
    sizeY = height / k;
    px    = 3 * NUM_SECTOR;  // px=3*9=27 
    p     = px;
	stringSize = sizeX * p; // stringSize = 27*sizeX  
    allocFeatureMapObject(map, sizeX, sizeY, p);
	// image：输入图像.  
	// dx：输出图像.  
	// kernel_dx：卷积核, 单通道浮点矩阵. 如果想要应用不同的核于不同的通道，先用 cvSplit 函数分解图像到单个色彩通道上，然后单独处理。  
	// cvPoint(-1, 0)：核的锚点表示一个被滤波的点在核内的位置。 锚点应该处于核内部。缺省值 (-1,-1) 表示锚点在核中心。  
	// 函数 cvFilter2D 对图像进行线性滤波，支持 In-place 操作。当核运算部分超出输入图像时，函数从最近邻的图像内部象素差值得到边界外面的象素值。  
    cvFilter2D(image, dx, &kernel_dx, cvPoint(-1, 0));// 起点在(x-1,y)，按x方向滤波
	cvFilter2D(image, dy, &kernel_dy, cvPoint(0, -1)); // 起点在(x,y-1)，按y方向滤波

	// 初始化cos和sin函数
    float arg_vector;
	// 计算梯度角的边界，并存储在boundary__y中
    for(i = 0; i <= NUM_SECTOR; i++)
    {
        arg_vector    = ( (float) i ) * ( (float)(PI) / (float)(NUM_SECTOR) );// 每个角的角度
        boundary_x[i] = cosf(arg_vector);// 每个角度对应的余弦值
        boundary_y[i] = sinf(arg_vector);// 每个角度对应的正弦值
    }/*for(i = 0; i <= NUM_SECTOR; i++) */

    r    = (float *)malloc( sizeof(float) * (width * height));
    alfa = (int   *)malloc( sizeof(int  ) * (width * height * 2));

    for(j = 1; j < height - 1; j++)
    {

		                                                                 // 每一行起点 (首地址） 
        datadx = (float*)(dx->imageData + dx->widthStep * j);
        datady = (float*)(dy->imageData + dy->widthStep * j);
        for(i = 1; i < width - 1; i++)                                  // 遍历一行中的非边界像素
        {
			                                                            // 遍历该行每一个元素 
            c = 0;                                                      // 第一颜色通道
            x = (datadx[i * numChannels + c]);
            y = (datady[i * numChannels + c]);

            r[j * width + i] =sqrtf(x * x + y * y);                    // 计算0通道的梯度大小

			// 使用向量大小最大的通道替代储存值
            for(ch = 1; ch < numChannels; ch++)
            {
                tx = (datadx[i * numChannels + ch]);
                ty = (datady[i * numChannels + ch]);
                magnitude = sqrtf(tx * tx + ty * ty);                   // 计算幅值 
                if(magnitude > r[j * width + i])                        // 找出每个像素点的梯度的最大值（有三个颜色空间对应的梯度），并记录通道数以及水平梯度以及垂直梯度
                {
                    r[j * width + i] = magnitude;                       // r表示最大幅值
                    c = ch;                                             // c表示这个幅值来自的通道序号
                    x = tx;                                             // x表示这个幅值对应的坐标处的x梯度
                    y = ty;                                             // y表示这个幅值对应的坐标处的y梯度
                }
            }/*for(ch = 1; ch < numChannels; ch++)*/

			// 使用sqrt（cos*x*cos*x+sin*y*sin*y）最大的替换掉 
            max  = boundary_x[0] * x + boundary_y[0] * y;
            maxi = 0;

			// 假设像素点的梯度方向为a,梯度方向为t,梯度大小为r,则dotProd=r*cosa*cost+r*sina*sint=r*cos(a-t)
            for (kk = 0; kk < NUM_SECTOR; kk++)                        // 遍历9个HOG划分的角度范围
            {
                dotProd = boundary_x[kk] * x + boundary_y[kk] * y;     // 计算两个向量的点乘
			
				// 若dotProd最大，则说明t最接近a
                if (dotProd > max) 
                {
                    max  = dotProd;
                    maxi = kk;
                }
				// 若-dotProd最大，则说明t最接近a+pi
                else 
                {
                    if (-dotProd > max) 
                    {
                        max  = -dotProd;
                        maxi = kk + NUM_SECTOR;       // 周期的，所以+一个周期NUM_SECTOR  
                    }
                }
            }
			//储存cos和sin的周期值
            alfa[j * width * 2 + i * 2    ] = maxi % NUM_SECTOR;
            alfa[j * width * 2 + i * 2 + 1] = maxi;  
        }/*for(i = 0; i < width; i++)*/
    }/*for(j = 0; j < height; j++)*/

    nearest = (int  *)malloc(sizeof(int  ) *  k);
    w       = (float*)malloc(sizeof(float) * (k * 2));
	// 给nearest初始化，为了方便以后利用相邻的cell的特征计算block（8*8，每个block以一个cell为中心，以半个cell为边界厚度）的属性
    for(i = 0; i < k / 2; i++)
    {
        nearest[i] = -1;
    }/*for(i = 0; i < k / 2; i++)*/
    for(i = k / 2; i < k; i++)
    {
        nearest[i] = 1;
    }/*for(i = k / 2; i < k; i++)*/
	//给w初始化，可能是cell（4*4）中每个像素贡献给直方图的权值（1/8+3/8+5/8+7/8+7/8+5/8+3/8+1/8）*（1/8+3/8+5/8+7/8+7/8+5/8+3/8+1/8）=4*4
    for(j = 0; j < k / 2; j++)
    {
        b_x = k / 2 + j + 0.5f;
        a_x = k / 2 - j - 0.5f;
        w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
        w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
    }/*for(j = 0; j < k / 2; j++)*/
    for(j = k / 2; j < k; j++)
    {
        a_x = j - k / 2 + 0.5f;
        b_x =-j + k / 2 - 0.5f + k;
        w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
        w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
    }/*for(j = k / 2; j < k; j++)*/

    for(i = 0; i < sizeY; i++)
    {
      for(j = 0; j < sizeX; j++)
      {
        for(ii = 0; ii < k; ii++)
        {
          for(jj = 0; jj < k; jj++)
          {
	   //第i行的第j个cell的第ii行第jj个像素
            if ((i * k + ii > 0) && 
                (i * k + ii < height - 1) && 
                (j * k + jj > 0) && 
                (j * k + jj < width  - 1))                 //要跳过厚度为1的边界像素，因为边界的梯度值不准确，但这样会导致含有边界的cell统计不完整
            {
              d = (k * i + ii) * width + (j * k + jj);
              (*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += 
                  r[d] * w[ii * 2] * w[jj * 2];            //第i行第j个cell的第alfa[d * 2]个梯度方向（0-8）
              (*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2];            //第i行第j个cell的第alfa[d * 2+1]个梯度方向（9-26）
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1))
              {
                (*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
                (*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
              }
              if ((j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                (*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
                (*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
              }
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1) && 
                  (j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
              }
            }
          }/*for(jj = 0; jj < k; jj++)*/
        }/*for(ii = 0; ii < k; ii++)*/
      }/*for(j = 1; j < sizeX - 1; j++)*/
    }/*for(i = 1; i < sizeY - 1; i++)*/
    
    cvReleaseImage(&dx);
    cvReleaseImage(&dy);


    free(w);
    free(nearest);
    
    free(r);
    free(alfa);

    return LATENT_SVM_OK;
}

/*****************************************************************************
// Feature map Normalization and Truncation 
//
// API
// int normalizeAndTruncate(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
函数功能：特征图标准化与截断（Feature map Normalization and Truncation）
函数参数：特征图，截断阈值
函数输出：标准化与截断之后的特征图
RESULT：Error status
*****************************************************************************/

int normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, const float alfa)
{
	//计算步骤：
	//1.分别计算每个block（除去边界）的9分特性的9个特性的平方和
	//2.分别计算每个block在各个方向上的9分特性的2范数
	//3.用各个属性（共27个）除以各个方向上的2范数，得到归一化的27*4个属性

    int i,j, ii;
    int sizeX, sizeY, p, pos, pp, xp, pos1, pos2;
    float * partOfNorm;                 // norm of C(i, j)
    float * newData;
    float   valOfNorm;                  //大小为block的总数，计算每个block的前九个特征的2范数
	// 初始化Hog所需要的参数  
    sizeX     = map->sizeX;
    sizeY     = map->sizeY;
    partOfNorm = (float *)malloc (sizeof(float) * (sizeX * sizeY));

    p  = NUM_SECTOR;                   //每个cell的bin的数目
    xp = NUM_SECTOR * 3;               //每个block的总特征数（9+18）
    pp = NUM_SECTOR * 12;

    for(i = 0; i < sizeX * sizeY; i++)
    {
        valOfNorm = 0.0f;
        pos = i * map->numFeatures;    //第i个block的第一个特征点索引号
        for(j = 0; j < p; j++)
        {
            valOfNorm += map->map[pos + j] * map->map[pos + j];         //计算第i个block的前9个特征的平方和
        }/*for(j = 0; j < p; j++)*/
        partOfNorm[i] = valOfNorm;
    }/*for(i = 0; i < sizeX * sizeY; i++)*/
    
    sizeX -= 2;                         //去掉第一列和最后一列的block
    sizeY -= 2;                          //去掉一第行和最后一行的block
	// 新建一个map->map的指针 
    newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));
//normalization
    for(i = 1; i <= sizeY; i++)
    {
        for(j = 1; j <= sizeX; j++)
        {
			//右下
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON;          //计算该block右下四个block的9分属性的2范数  
            pos1 = (i  ) * (sizeX + 2) * xp + (j  ) * xp;                           //第i行第j列的block的属性的第一个值的索引值
            pos2 = (i-1) * (sizeX    ) * pp + (j-1) * pp;                            //除掉边框后的第i-1行第j-1列的block的newdata的首地址   
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii        ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 4] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/

			//右上
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON;
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p    ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 6] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/

			//左下
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON;
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p * 2] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 8] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/

			//左上
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON;
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p * 3 ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 10] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
        }/*for(j = 1; j <= sizeX; j++)*/
    }/*for(i = 1; i <= sizeY; i++)*/
//truncation
    for(i = 0; i < sizeX * sizeY * pp; i++)
    {
        if(newData [i] > alfa) newData [i] = alfa;
    }/*for(i = 0; i < sizeX * sizeY * pp; i++)*/
//swop data
	// 将计算结果，指针复制到结果输出的map上  
    map->numFeatures  = pp;
    map->sizeX = sizeX;
    map->sizeY = sizeY;

    free (map->map);
    free (partOfNorm);

    map->map = newData;

	return LATENT_SVM_OK; // return 0  
}

/*****************************************************************************
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
函数功能：特征图降维（Feature map reduction）
In each cell we reduce dimension of the feature vector according to original paper special procedure
函数参数：特征图
函数输出：特征图
RESULT：Error status
*****************************************************************************/

int PCAFeatureMaps(CvLSVMFeatureMapCaskade *map)
{ 
	//步骤：
	//1.计算每个18分属性在4个方向上的和；
	//2.计算每个9分属性在4个方向上的和
	//3.计算4个方向上18分属性的和

    int i,j, ii, jj, k;
    int sizeX, sizeY, p,  pp, xp, yp, pos1, pos2;
    float * newData;
    float val;
    float nx, ny;
    
    sizeX = map->sizeX;
    sizeY = map->sizeY;
    p     = map->numFeatures;
    pp    = NUM_SECTOR * 3 + 4;
    yp    = 4;
    xp    = NUM_SECTOR;

    nx    = 1.0f / sqrtf((float)(xp * 2));
    ny    = 1.0f / sqrtf((float)(yp    ));

    newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));

    for(i = 0; i < sizeY; i++)
    {
        for(j = 0; j < sizeX; j++)
        {
            pos1 = ((i)*sizeX + j)*p;        //去掉边界后的第i行第j列的block的的第一个属性值的索引值
            pos2 = ((i)*sizeX + j)*pp;       //newData关于第i行第j列的block的的第一个属性值的索引值
            k = 0;
            for(jj = 0; jj < xp * 2; jj++)
            {
                val = 0;
                for(ii = 0; ii < yp; ii++)
                {
                    val += map->map[pos1 + yp * xp + ii * xp * 2 + jj]; //计算每个block的18分属性在四个方向的和
                }/*for(ii = 0; ii < yp; ii++)*/
                newData[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp * 2; jj++)*/
            for(jj = 0; jj < xp; jj++)
            {
                val = 0;
                for(ii = 0; ii < yp; ii++)
                {
                    val += map->map[pos1 + ii * xp + jj];
                }/*for(ii = 0; ii < yp; ii++)*/
                newData[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp; jj++)*/
            for(ii = 0; ii < yp; ii++) //9分属性
            {
                val = 0;
                for(jj = 0; jj < 2 * xp; jj++)
                {
                    val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];    //计算每个block的18分属性在一个方向上的和，
                }/*for(jj = 0; jj < xp; jj++)*/
                newData[pos2 + k] = val * nx;
                k++;
            } /*for(ii = 0; ii < yp; ii++)*/           
        }/*for(j = 0; j < sizeX; j++)*/
    }/*for(i = 0; i < sizeY; i++)*/
//swop data
	// 将计算结果，指针复制到结果输出的map上  
    map->numFeatures = pp;

    free (map->map);

    map->map = newData;

    return LATENT_SVM_OK;
}


//modified from "lsvmc_routine.cpp"
// 根据输入，转换成指针**obj，其中(*obj)->map为sizeX * sizeY  * numFeatures大小 
int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, 
                          const int sizeY, const int numFeatures)
{
    int i;
    (*obj) = (CvLSVMFeatureMapCaskade *)malloc(sizeof(CvLSVMFeatureMapCaskade));
    (*obj)->sizeX       = sizeX;
    (*obj)->sizeY       = sizeY;
    (*obj)->numFeatures = numFeatures;
    (*obj)->map = (float *) malloc(sizeof (float) * 
                                  (sizeX * sizeY  * numFeatures));
    for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    {
        (*obj)->map[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
// 释放自己定义的CvLSVMFeatureMapCaskade数据 
int freeFeatureMapObject (CvLSVMFeatureMapCaskade **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}
