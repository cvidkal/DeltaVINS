#pragma once

#if USE_ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <nav_msgs/msg/path.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "FrameAdapter.h"
#include "WorldPointAdapter.h"

namespace DeltaVins {
class DataOutputROS : public rclcpp::Node,
                      public WorldPointAdapter,
                      public FrameAdapter {
   public:
    using Ptr = std::shared_ptr<DataOutputROS>;
    DataOutputROS();
    ~DataOutputROS();

    void PushViewMatrix(std::vector<FrameGL> &v_Rcw) override;

    void PushImageTexture(unsigned char *imageTexture, const int width,
                          const int height, const int channels,
                          const std::string &name) override;

    void PushWorldPoint(const std::vector<WorldPointGL> &v_Point3f) override;

    void FinishFrame() override {}

   private:
    std::unordered_map<std::string,
                       rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
        image_publishers_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        point_cloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        current_point_cloud_publisher_;
    std::unordered_map<int, WorldPointGL> world_points_;

    void CreateImagePublisher(const std::string &name);
};
}  // namespace DeltaVins

#endif