#include "image_filtering.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


uchar adaptiveProcess(const Mat &im,int row,int col,int kernelSize,int maxSize)
{
    vector<uchar>pixels;
    for(int a = -kernelSize /2;a<=kernelSize /2;a++)
    {
        for(int b=-kernelSize/2;b<=kernelSize/2;b++)
        {
            pixels.push_back(im.at<uchar>(row + a,col +b));
        }
    }
    sort(pixels.begin(),pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize*kernelSize - 1];
    auto med = pixels[kernelSize*kernelSize / 2];
    auto zxy = im.at<uchar>(row,col);
    if(med > min && med <max)
    {
        //med大于min且小于max，则说明中值不是一个噪声，这时候如果当前kernel中的
        //中心点也不是噪声，就返回中心点的值，不做滤波。如果中心点是噪声，才返回中值滤波的值
        if(zxy > min && zxy <max)
        {
            return zxy;
        }
        else
        {
            return med;
        }
    }
    //如果中值是一个噪声，则扩大窗口，寻找不是噪声的中值来替代。
    else
    {
        
        kernelSize +=2;
        if(kernelSize <=maxSize)
        {
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
        }
        else
        {
            return med;
        }
    }
}
/**
 *自适应中值滤波：中值滤波器的效果受滤波窗口尺寸的影响较大，在消除噪声和保护图像的细节存在着矛盾：滤波窗口较小，则能很好的保护图像中的某些细节，但对噪声的过滤效果就不是很好；反之，窗口尺寸较大有较好的噪声过滤效果，但是会对图像造成一定的模糊。另外，根据中值滤波常规的中值滤波器的窗口尺寸是固定大小不变的，就不能同时兼顾去噪和保护图像的细节。这时就要寻求一种改变，根据预先设定好的条件，在滤波的过程中，动态的改变滤波器的窗口尺寸大小，这就是自适应中值滤波器 Adaptive Median Filter。在滤波的过程中，自适应中值滤波器会根据预先设定好的条件，改变滤波窗口的尺寸大小，同时还会根据一定的条件判断当前像素是不是噪声，如果是则用邻域中值替换掉当前像素；不是，则不作改变。器原理，如果在滤波窗口内的噪声点的个数大于整个窗口内像素的个数，则中值滤波就不能很好的过滤掉噪声。
 *@param1:输入图像src
 *@param2:输出图像dst
 *@param3:minSize为最小的kernelsize
 *@param4: maxSize为最大的kernlesize
 */
void adpativeMeanFilter(const Mat&src,Mat&dst,int minSize= 3,int maxSize = 7)
{
    //先按照最大的kernelsize来makeborder，以防万一
    copyMakeBorder(src, dst, maxSize/2, maxSize/2, maxSize/2, maxSize/2, BORDER_REFLECT);
    int rows = dst.rows;
    int cols = dst.cols;
    
    for(int j = maxSize /2;j<rows - maxSize /2;++j)
    {
        for(int i = maxSize/2;i<cols * dst.channels() - maxSize/2;++i)
        {
            //从最小的kernelsize开始滤波
            dst.at<uchar>(j,i) = adaptiveProcess(dst, j, i, minSize, maxSize);
        }
    }
}

int main()
{
    Mat src = imread("hehua.jpg");
    Mat dst;
    adpativeMeanFilter(src, dst);
    imshow("src",src);
    imshow("dst",dst);
    waitKey(0);
    return 0;
}
