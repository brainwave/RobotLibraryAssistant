//Standard header files
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
using namespace std;
using namespace cv;

//Global Variables
Mat src_1;
Mat src_2;
Mat src_new;

Mat src_orig;
Mat src_trans;
char* window_name="Image";


//SURF Extraction, with Flann Matcher
void surfAlgo (int, void*)
{
	int key=0, i=0;
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *imageKeyPoints=0, *imageDescriptors = 0;
	vector<KeyPoint> keyPoints_orig, keyPoints_trans;
	Mat descriptor_orig, descriptor_trans;
	double max_dist=0;double min_dist=100;
	std::vector < DMatch > good_matches;
	
	FlannBasedMatcher matcher;
	std::vector<DMatch>matches;

	SurfFeatureDetector detector(1500);
	SurfDescriptorExtractor extractor;

	cvtColor (src_2,src_trans,CV_BGR2GRAY);
	cvtColor (src_1,src_orig,CV_BGR2GRAY);

	detector.detect(src_orig, keyPoints_orig);
	detector.detect(src_trans, keyPoints_trans);

	extractor.compute(src_orig, keyPoints_orig, descriptor_orig);
	extractor.compute(src_trans, keyPoints_trans, descriptor_trans);
	
	matcher.match(descriptor_orig, descriptor_trans, matches);
	
	for (int count=0;count<descriptor_orig.rows;count++)
	{
		double dist=matches[i].distance;
		if( dist < min_dist ) min_dist=dist;
		if (dist > max_dist) max_dist = dist;
	}

	
	for(int count = 0; count < descriptor_orig.rows; count++)
	{
		if( matches[count].distance < 3*min_dist )
			{
				good_matches.push_back( matches[count]);
			}
	}

	//drawKeypoints(src_trans, keyPoints_trans, src_trans, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(src_orig, keyPoints_orig, src_orig, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);	
	
	//drawMatches ( src_orig, keyPoints_orig, src_trans, keyPoints_trans, matches, src_trans);

	drawMatches( src_orig, keyPoints_orig, src_trans, keyPoints_trans,good_matches,src_new,Scalar(0,0,255),Scalar(0,0,255),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int count=0; count<good_matches.size();count++)
	{
		obj.push_back(keyPoints_trans[ matches[count].queryIdx].pt);
		scene.push_back(keyPoints_orig[ matches[count].trainIdx].pt);
	}

/*	
 	Code work in progress

	Mat H = findHomography( obj, scene, CV_RANSAC );
	std::vector<Point2f> object_corners(4);
  	std::vector<Point2f> scene_corners(4);


	object_corners[0] = cvPoint (0,0);
       	object_corners[1] = cvPoint (src_trans.cols,0);
	object_corners[2]= cvPoint (src_trans.cols,src_trans.rows);
	object_corners[3] = cvPoint (0,src_trans.rows);

	perspectiveTransform( object_corners, scene_corners, H);

	line (src_new, scene_corners[0]+Point2f(src_trans.cols,0), scene_corners[1]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[1]+Point2f(src_trans.cols,0), scene_corners[2]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[2]+Point2f(src_trans.cols,0), scene_corners[3]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[3]+Point2f(src_trans.cols,0), scene_corners[0]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
*/
	imwrite("/Users/brainwave/Desktop/testrun1.jpg",src_new);
	resize(src_new,src_new,Size(),0.3,0.3,CV_INTER_LINEAR);

	imshow(window_name, src_new);

}	

int main(int argc, char** argv)
	
{
		
	VideoCapture vidcap= VideoCapture(argv[1]);
	cvNamedWindow (window_name,CV_WINDOW_AUTOSIZE); 
	
	src_1=imread(argv[2]);
	src_2=imread(argv[1]);
	surfAlgo(0,0);	

	waitKey(0);	
	return 0;
}


