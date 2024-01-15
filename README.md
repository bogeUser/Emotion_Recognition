## Emotion_Recognition

面部识别

安装：
    环境： Windows10 + python3.7.4
    dlib==19.21.0
    opencv-python==4.3.0.25
    numpy==1.16.4
    imutils==0.5.2
    keras==2.2.4
    scipy==1.4.1
    tensorflow==1.13.1


folder
    algorithms: 算法包
        coordinate2points.py: 提取人脸128个关键点的特征值
        emotion_recognition.py: 情绪识别核心算法模块
        face.py:  对当前的人脸做出判断前的预处理模块
        utils.py: 进行人脸数据计算的相关工具包
    camera: 封装的摄像机模块
        camera.py: 封装的摄像机类，包含一个拉流方法，一个显示姓名的方法（兼容中文字符），一个显示情绪名称的方法，一个显示头部姿态的方法
    config: 配置文件模块
        config.py: 包含所有算法模型的文件地址，以及后续需要做进一步处理时的相关配置，目前只用到了算法模型的文件地址
    image: 图片包
        face_database:  临时图片文件数据库
        test_img: 测试图片库
    log: 日志模块
    main: 主函数模块，考虑到程序运行时对硬件资源的使用，将三个功能处理成三个进程，一个人脸身份识别并显示姓名，一个情绪识别，
          一个头部姿态。如果可同时运行可以分开单独运行。
    model: 算法模型，网络模型
    save: 截图/视频存储模块, 暂未使用

运行步骤：
    1.直接运行main.py 文件。 该文件的run方法中有三个进程，可以同时运行三个进程，也可以只运行其中一个。如果指定运行其中一个进程，把
      另外两个进程注释就可以了。
    2.face_database文件夹中可以自行添加图片，用于静态图片的算法测试。图片命名可以是中文，也可以是英文。图片
      格式必须是jpg格式
    3.用于测试的图片放在test_img 文件中， 图片格式必须为jpg格式。

运行结果：
    进程draw_name中红色框表示人脸数据库中没有该人脸，绿色框表示人脸数据库中有该人脸，并且给出了该对象的人名
    进程draw_emotion人脸用绿色框表示，并且给出每张人脸的表情
    进程draw_state在人脸没有进行画框，在每个人脸上给出了该人的头部实时状态信息， 其中Camera类的eye_ar_thresh=0.25,
                 eye_ar_consec_frames=3, mar_thresh=0.8, mouth_arconsec_frames=2, har_thresh=0.3,
                 nod_ar_consec_frames=2, par_thresh=25, par_duration=3 参数为默认值，在测试的时候可调。

注意事项： 1，人脸识别，情绪识别，头部姿态识别分别使用三个进程运行， 三个进行可单独运行，也可同时运行。
         2，算法准确度和效率，根据实际应用场景进行优化和改进。