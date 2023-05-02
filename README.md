# OpenCV 部署快速风格迁移

效果：

![](./images/out.jpg)

## 1. 快速风格迁移简介

[查看我的博客](https://blog.alexsun.top/vuepress-opencv-notes/awesome/fast-neural-style/) 阅读更多关于快速风格迁移的内容。

[*Perceptual Losses for Real-Time Style Transfer and Super-Resolution*](https://arxiv.org/abs/1603.08155) 的实现，在 [此项目的官方网站](https://cs.stanford.edu/people/jcjohns/eccv16/) 上，可以查看论文和其效果，推荐阅读。

此项目的原始实现（Lua）可以参考 [jcjohnson/fast-neural-style](https://github.com/jcjohnson/fast-neural-style)，可以下载其预训练权重进行部署。

## 2. 下载 Torch 模型文件

下载完成后保存到：

- `models/`
    - `eccv16/`
        - [`composition_vii.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/composition_vii.t7)
        - [`la_muse.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/la_muse.t7)
        - [`starry_night.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7)
        - [`the_wave.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/the_wave.t7)
    - `instance_norm/`
        - [`candy.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7)
        - [`feathers.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7)
        - [`la_muse.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7)
        - [`mosaic.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7)
        - [`the_scream.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7)
        - [`udnie.t7`](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7)

## 3. Python 版本

需要 Python >= 3.7，依赖：
- `opencv-python>=4.5`，建议自行编译 CUDA 版本
- `numpy>=1.19`

运行 Python 脚本：

```bash
python python/main.py models/eccv16/starry_night.t7
```

## 4. C++ 版本

### 4.1 必要条件

需要安装的依赖：
- Visual Studio 2019 / 2022
- OpenCV >= 4.5，建议自行编译 CUDA 版本
- CMake

需要安装的 VS Code 插件：
- C/C++
- CMake

### 4.2 配置 VS Code

这里是我的配置项目，请根据自己的配置调整。使用 Visual Studio 2022 [Release] x64 - amd64，在 `.vscode/settings.json` 配置：

```json
{
    "cmake.configureEnvironment": {
        "OpenCV_DIR": "D:/workspace/repo/opencv4.7/opencv-4.7.0/build/install"
    }
}
```

然后在 `.vscode/c_cpp_properties.json` 中配置：

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${default}",
                "${workspaceFolder}/**",
                "D:/workspace/repo/opencv4.7/opencv-4.7.0/build/install/include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.19044.0",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-msvc-x64",
            "configurationProvider": "ms-vscode.cmake-tools",
            "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/bin/Hostx64/x64/cl.exe"
        }
    ],
    "version": 4
}
```

这一步需要根据你安装的 Windows SDK 和 MSVC 版本进行调整。

### 4.3 运行

首先，使用 F1 打开 VS Code 命令面板，输入 `CMake: Configure`，选择 `x64-Release`，等待 CMake 完成配置。

然后在下方点击运行即可，如果出现没有找到模型的错误，可以执行：

```bash
cd ../..
```

切换到项目根目录，然后再次运行。

## 5. License | 开源许可

MIT License.
