/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"


#include<cstring>
#include "IOWrapper/OpenCV/ImageRW_OpenCV.cpp"
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
using namespace cv;



int counter=1;


namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class SampleOutputWrapper : public Output3DWrapper
{
public:
        inline SampleOutputWrapper()
        {
            num_of_point_cloud = 0;
            save_point_cloud = true;
            is_point_cloud_file_close = false;            
			point_cloud_file_obj.open(point_cloud_file_name);
            printf("OUT: Created SampleOutputWrapper\n");
        }

        virtual ~SampleOutputWrapper()
        {
            point_cloud_file_obj.close();
            printf("OUT: Destroyed SampleOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            /*
            printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
                printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
                maxWrite--;
                if(maxWrite==0) break;
            }
            */
        }


        /*
            Goal: Save point cloud from Direct Sparse Odometry (DSO)
            Credit: https://github.com/Neoplanetz/dso_with_saving_pcl 
        */

        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {

            float FX,FY; // focal length expressed in pixels and it translate a pixel coordinate into lengths
            float CX,CY; // coordinates of the principal points
            float FXi, FYi, CXi, CYi;
            //fxl(), fyl(), cxl(), cyl(): get optimized, most recent (pinhole) camera intrinsics.
            FX = HCalib->fxl(); // get focal length
            FY = HCalib->fyl(); // get focal length
            CX = HCalib->cxl(); // get principal point or center of the image
            CY = HCalib->cyl(); // get principal point or center of the image
            
            FXi = 1 / FX;
            FYi = 1 / FY;
            CXi = -CX / FX;
            CYi = -CY / FY;

            if (final==true) { //Get the final model, but don't care about it being delay-free and to save compute
                for (FrameHessian* f : frames) {
                    bool is_pose_valid=f->shell->poseValid; //>poseValid = false if [camToWorld] is invalid (only happens for frames during initialization)
  
                    if (is_pose_valid) { // if valid
                        auto const& m = f->shell->camToWorld.matrix3x4();   //describes the mapping of a pinhole camera from 3D points in the world to 2D points in an image     
                        auto const& points = f->pointHessiansMarginalized; //contains marginalized points.
                        for (auto const* p : points) {
                            float depth = (1.0f/p->idepth); // convert the inverse depth into depth
                            // convert depth map into point cloud
                            // calculate the value of x,y,z coordinates
                            // idea behind this: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
                            /*
                                Image plane in p->u, p->v coordinates
                                Each pixel has a colour and a depth value
                                On the otherhand x,y,z are cartesian coordinates
                            */
                            auto const x = (p->u * FXi + CXi) * depth;
                            auto const y = (p->v * FYi + CYi) * depth;
                            auto const z = depth * (1 + 2 * FXi);

                            Eigen::Vector4d camPoint(x, y, z, 1.f);
                            Eigen::Vector3d real_world_coordinates = m * camPoint;

                            // save point cloud when it is save mode
                            if (save_point_cloud && point_cloud_file_obj.is_open()) { // save point cloud
                                write_point_cloud = true;
                                point_cloud_file_obj << real_world_coordinates[0] << " " << real_world_coordinates[1] << " " << real_world_coordinates[2] << "\n";
                                num_of_point_cloud++;
                                write_point_cloud = false;
                            }
                            // otherwise close the point cloud file object
                            else {
                                if (!is_point_cloud_file_close) {
                                    if (point_cloud_file.is_open()) {
                                        point_cloud_file_obj.flush();
                                        point_cloud_file_obj.close();
                                        is_point_cloud_file_close = true;
                                    }
                                }
                            }


                         }
                    }
                }
            }


        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            /*
            printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
                   frame->incoming_id,
                   frame->timestamp,
                   frame->id);
            std::cout << frame->camToWorld.matrix3x4() << "\n";
            */
        }


        virtual void pushLiveFrame(FrameHessian* image) override
        {
            // can be used to get the raw image / intensity pyramid.
        }

       /*
            Store semi dense depth
       */
       virtual void pushDepthImage(MinimalImageB3* image) override
        {   
            //writeImage("/home/rafiqul/results/"+boost::to_string(counter)+".png",image); //store semi dense depth
            //counter++;
        }
        virtual bool needPushDepthImage() override
        {
            return true;
        }


        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
            printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
                   KF->frameID,
                   KF->shell->incoming_id,
                   KF->shell->timestamp,
                   KF->shell->id);
            std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            Mat cv_img(image->w,image->h,CV_8UC3);
            //eigen2cv(image, img);
            
            for(int y=0;y<image->h;y++)
            {
                for(int x=0;x<image->w;x++)
                {
                    if(image->at(x,y) <= 0) continue;

                    printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    cv_img.at<float>(x,y)=image->at(x,y);
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
            //writeImage("/home/rafiqul/results/"+boost::to_string(KF->frameID)+".png",image);
            //imwrite("/home/rafiqul/results/"+boost::to_string(KF->frameID)+".png",cv_img);
            //counter++;

        }

        std::ofstream point_cloud_file_obj;


};



}



}