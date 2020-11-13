/*
KCFTracker类中函数的定义
Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif
#include <iostream>
#include <iomanip>
using namespace std;
// Constructor   初始化KCF类参数
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5; 
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;


    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;           //线性插值因子的适应
        sigma = 0.6;                    //高斯核带宽
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5; 
        cell_size = 4;
        _hogfeatures = true;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4; 
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;                  //lab质心,,将其初始化为 nClusters(聚类)行,3列,类型为 CV_32FC1,数据全部初始化为data
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2; 
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale 模板尺寸
        template_size = 96;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale  拟合校正
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}

/*****************************************************************************
*函数功能：初始化跟踪器，包括回归参数的计算，变量的初始化（Initialize tracker）
*函数参数：目标初始框的引用，初始帧、
*****************************************************************************/

// Initialize tracker 使用第一帧和它的跟踪框，初始化KCF跟踪器
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;                                           //_roi是基类Tracker的protected成员变量   
    assert(roi.width >= 0 && roi.height >= 0);          
    _tmpl = getFeatures(image, 1);      // 获取特征，在train里面每帧修改  
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);  //这个不修改了，只初始化一次
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));  //获取特征，在train里面每帧修改 
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
 }

/*****************************************************************************
*函数功能：获取当前帧的目标位置以及尺度（Update position based on the new frame）
*函数参数：当前帧的整幅图像
*****************************************************************************/
// Update position based on the new frame  基于当前帧更新目标位置  
cv::Rect KCFTracker::update(cv::Mat image)
{
	//修正边界
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;


    // 跟踪框中心 
    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;
	cout << setiosflags(ios::left | ios::fixed);    // 设左对齐，以固定小数位显示
	cout.precision(3);
//	cout << "X坐标:"<<cx<<"  "<< "Y的坐标："<< cy <<endl;
    //cout << cx << endl;
	//cout << "y坐标:" << endl;
	//cout.precision(4);
	//cout << cy << endl;

//	cout << "左上X坐标:" << _roi.x << "  " << "左上角Y坐标：" << _roi.y << "  " << "右上角X坐标:：" << _roi.x + _roi.width << "  " << "右上Y角坐标:：" << _roi.y << endl;
//	cout << "左下X坐标:" << _roi.x << "  " << "左下角Y坐标：" << _roi.y + _roi.height << "  " << "右下角X坐标:：" << _roi.x + _roi.width << "  " << "右下Y角坐标:：" << _roi.y + _roi.height << endl;


	// 尺度不变时检测峰值结果 
    float peak_value;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);

	// 略大尺度和略小尺度进行检测 
    if (scale_step != 1) {
        // Test at a smaller _scale  使用一个小点的尺度测试  
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);
		
		// 做减益还比同尺度大就认为是目标 
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }

        // Test at a bigger _scale
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }

    // Adjust by cell size and _scale  因为返回的只有中心坐标，使用尺度和中心坐标调整目标框 
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);
	// 使用当前的检测框来训练样本参数  
    cv::Mat x = getFeatures(image, 0);
    train(x, interp_factor);

	return _roi; //返回检测框  
}


/*****************************************************************************
*函数功能：根据上一帧结果计算当前帧的目标位置（Detect object in the current frame）
*函数参数：之前训练（初始化）的结果，当前的特征图，当前最高得分（引用）
*****************************************************************************/
// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)   // z为前一帧样本  ,x为当前帧图像,peak_value为输出的峰值 
{
    using namespace FFTTools;                                             // 做变换得到计算结果res 

    cv::Mat k = gaussianCorrelation(x, z);                                // 计算x和z之间的高斯相关核(公式)
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates  使用opencv的minMaxLoc来定位峰值坐标位置 
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer 
	// 子像素峰值检测，坐标是非整形的 
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

/**************************************************************************
*函数功能：根据每一帧的结果训练样本并更新模板（train tracker with a single image）
*函数参数：新的目标图像，训练因子train_interp_factor是interp_factor
*****************************************************************************/
// train tracker with a single image
// 使用图像进行训练，得到当前帧的_tmpl，_alphaf 
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));                            //计算岭回归系数(公式)
    

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;                 // 更新模板的特征（公式）
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;        // 更新岭回归系数的值（公式）


    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}




/*****************************************************************************
*函数功能：使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移
必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）
Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window）
*函数参数：高斯核的两个参数
*****************************************************************************/
// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
// 使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移 
// 必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）  
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);                                     // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);               // 将第i个属性排列成原来cell的排列形式
            x2aux = x2.row(i).reshape(1, size_patch[0]);

			// 两个傅立叶频谱的每个元素的乘法 相乘-频谱
			// 输入数组1、输入数组2、输出数组(和输入数组有相同的类型和大小)
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d; 
	cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}






// Create Gaussian Peak. Function called only in the first frame.// 创建高斯峰函数，函数只在第一帧的时候执行 
/*****************************************************************************
*函数功能：创建高斯峰函数，仅在第一帧时被执行
*Create Gaussian Peak. Function called only in the first frame
*函数参数：二维高斯峰的X、Y的大小
*****************************************************************************/
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}


// Obtain sub-window from image, with replication-padding and extract features// 从图像得到子窗口，通过赋值填充并检测特征 
/*****************************************************************************
*函数功能：提取目标窗口的特征（Obtain sub-window from image, with replication-padding and extract features）
*函数参数：图像，是否使用汉宁窗，尺度调整因子
*****************************************************************************/
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{

	//步骤：
	//1.根据给定的框框找到合适的框框
	//2.提取HOG特征
	//3.对特征进行归一化和截断
	//4.对特征进行降维
	//5.获取Lab特征,并将结果与hog特征进行连接
	//6.创建一个常数阵，对所有特征根据cell的位置进行加权

    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;
	// 初始化hanning窗， 其实只执行一次，只在第一帧的时候inithann=1  
    if (inithann) {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;
		// 按照长宽比例修改长宽大小，保证比较大的边为template_size大小  
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }
		// 设置_tmpl_sz的长宽：向上取原来长宽的最小2*cell_size倍 
		// 其中，较大边长为104 
        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        }
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }
	// 以上都是调整_tmpl_sz的大小为了各种适应


	// 检测区域大小  
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size
	// 检测区域左上角坐标  
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;
	// 提取目标区域像素，超边界则做填充 
    cv::Mat FeaturesMap;  
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	// 按照比例缩小边界大小
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    }   

    // HOG features // 提取HOG特征点 
    if (_hogfeatures) {
        IplImage z_ipl = z;                                             // 之前的图像类
		CvLSVMFeatureMapCaskade *map;                                   // 申请指针  
		getFeatureMaps(&z_ipl, cell_size, &map);                        // 给map进行赋值 
        normalizeAndTruncate(map,0.2f);                                 // 归一化 
		PCAFeatureMaps(map);                                            // 由HOG特征变为PCA-HOG 
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);

        // Lab features   带有Lab特征在一些跟踪环节效果并不好 
        if (_labfeatures) {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char*)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0]*size_patch[1], CV_32F, float(0));    //每一列是一个cell的nClusters个属性

            int cntCell = 0;                                                                              //代表的是除边界以外的cell的索引号
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
                for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){

					//遍历除边界以外的cell，第cy行第cx列的cell
					// Iterate through each pixel of cell (cX,cY)
					//对每个cell的每个像素的lab值进行根据_labCentroids进行分类，分类标准：欧氏距离的平方

                    // Iterate through each pixel of cell (cX,cY)
                    for(int y = cY; y < cY+cell_size; ++y){
                        for(int x = cX; x < cX+cell_size; ++x){
                            // Lab components for each pixel  //三个通道，分别代表LAB空间的三个值
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid  （质心，矩心）
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float*)(_labCentroids.data);
                            for(int k = 0; k < _labCentroids.rows; ++k){
                                float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                           + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) ) 
                                           + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );//好像是求的lab和某一个特定lab的距离
                                if(dist < minDist){
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ; 
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and add features to FeaturesMap
            size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab); //将根据lab空间计算得到的结果和通过hog特征计算的结果进行合并
        }
    }
    else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;  
    }
    
    if (inithann) {
        createHanningMats();               // 创建了一个和FeatureMap大小相关的常数Mat（sizeX*sizezY)*size_patch[2]
    }
    FeaturesMap = hann.mul(FeaturesMap);  // 点乘 
    return FeaturesMap;
}
    

/*****************************************************************************
*函数功能：初始化cosine window，仅在第一帧调用（Initialize Hanning window. Function called only in the first frame.）
*函数参数：无
*****************************************************************************/
// Initialize Hanning window. Function called only in the first frame.// 初始化hanning窗，只执行一次，使用opencv函数做的 
void KCFTracker::createHanningMats()
{   
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

/*****************************************************************************
*函数功能：对目标的位置插值，提高精度,计算一维亚像素峰值
*使用幅值做差来定位峰值的位置，返回的是需要改变的偏移量大小
*Calculate sub-pixel peak for one dimension
*函数参数：无
*****************************************************************************/
// Calculate sub-pixel peak for one dimension// 使用幅值做差来定位峰值的位置，返回的是需要改变的偏移量大小
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}
