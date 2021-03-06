# LPR

LPR是一个集成多个开源项目面向嵌入式车牌识别库，检测和校准部分以ncnn作为推理后端，识别部分以opencv作为推理后端，支持多种车牌检测和识别

# 特点
- 超轻量，嵌入式设备可实时识别，荣品3288平均识别速度约100ms
- 支持多种车牌识别，支持大角度车牌识别
- 基于端到端的车牌识别无需进行字符分割
- 识别率高,支持各种场地的车牌识别
# 项目构造说明
- 车牌检测集成了1MB轻量级车牌检测模型
- 车牌校准集成了MTCNN第三级
- 车牌识别集成了HyperLPR项目端对端车牌识别
# 构建及安装
1. **下载源代码**

        https://github.com/zouxiangxiang/LPR.git

2.  **环境准备**    
  - 安装opencv3.0及以上
  - 安装cmake3.0以上版本，支持c++11的c++编译器
  - 安装protobuf
  - 安装ncnn

3. **Linux安装编译** 
       
        mkdir build
        cd build
        cmake ..
        make install

# 可识别和待支持的车牌的类型
  
- [x] 单行蓝牌
- [x] 单行黄牌
- [x] 新能源车牌
- [x] 白色警用车牌
- [x] 使馆/港澳车牌
- [x] 教练车牌
# 参考
[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR "HyperLPR")  
[https://github.com/xiangweizeng/mobile-lpr](https://github.com/xiangweizeng/mobile-lpr "mobile-lpr")  
[https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB "Ultra-Light-Fast-Generic-Face-Detector-1MB")  
