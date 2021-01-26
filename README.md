#LPR
**LPR是一个集成多个开源项目面向嵌入式车牌识别库，检测和校准部分以ncnn作为推理后端，识别部分以opencv作为推理后端，支持多种车牌检测和识别。**
#特点
- **超轻量，嵌入式设备可实时识别，荣品3288平均识别速度约100ms**
- **支持多种车牌识别，支持大角度车牌识别**
- **基于端到端的车牌识别无需进行字符分割**
- **识别率高,支持各种场地的车牌识别**
#项目构造说明
- **车牌检测集成了1MB轻量级车牌检测模型**
- **车牌校准集成了MTCNN第三级**
- **车牌识别集成了HyperLPR项目端对端车牌识别**
# 构建及安装
1. **下载源代码**
  
        https://github.com/zouxiangxiang/LPR.git

2.  **环境准备**    
  - 安装ncnn

  
