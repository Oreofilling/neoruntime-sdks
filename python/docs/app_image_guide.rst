应用镜像制作与导入指南
========================

概述
----

本指南介绍如何在开发环境中制作应用 Docker 镜像，打包为 ``.neoapp`` 应用包，
并导入到 NeoRuntime 设备上运行。

完整流程包括：

#. 准备应用文件（Dockerfile、app.yaml、app.py）
#. 构建 Docker 镜像
#. 导出镜像并打包为 ``.neoapp`` 应用包
#. 通过 Web Console 应用安装向导导入，或使用命令行导入

.. tip::

   交付产物是单文件 ``.neoapp`` 应用包（tar.gz，内含 ``app.yaml`` 与
   ``image.tar``）。在 Web Console 上传一个 ``.neoapp`` 即可完成安装，
   服务端自动解出配置与镜像；也可以只上传裸镜像 tar，由向导表单生成配置。

.. _app_image_step1:

步骤 1: 准备应用文件
--------------------

创建应用目录并准备以下核心文件。``app.yaml`` 会随镜像一起打进 ``.neoapp``
应用包，上载后仍可在安装向导中微调；只上传裸镜像时才完全依赖向导表单生成配置。

创建应用目录
~~~~~~~~~~~~

.. code-block:: bash

   mkdir my-app && cd my-app

应用代码（app.py）
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """My Application"""

   import signal
   from neoruntime_ipc_sdk import InferenceClient, EventClient, DeviceClient, Config


   class MyApp:
       def __init__(self):
           self.running = True
           self.app_id = Config.get_app_id()

           self.inference = InferenceClient()
           self.events = EventClient()
           self.device = DeviceClient()

           signal.signal(signal.SIGINT, self.signal_handler)
           signal.signal(signal.SIGTERM, self.signal_handler)

       def signal_handler(self, signum, frame):
           self.running = False

       def run(self):
           try:
               for frame, result in self.inference.subscribe(
                   stream="main", model="person_v1", fps=10
               ):
                   if not self.running:
                       break
                   person_count = result.count_by_label("person")
                   if person_count > 0:
                       self.events.publish(f"app/{self.app_id}/detection", {
                           "count": person_count,
                       })
           except Exception as e:
               print(f"[{self.app_id}] Error: {e}")
           finally:
               self.inference.close()
               self.events.close()
               self.device.close()


   if __name__ == "__main__":
       MyApp().run()

Dockerfile
~~~~~~~~~~

.. code-block:: dockerfile

   FROM python:3.9-slim

   LABEL maintainer="your@email.com"

   # 安装 NeoRuntime Python SDK（已发布到 PyPI；锁定版本保证可重复构建，
   # 升级 SDK 时同步修改此处版本号）
   RUN python -m pip install --no-cache-dir \
       "neoruntime-ipc-sdk==0.7.4"

   WORKDIR /app
   COPY app.py app.yaml /app/

   RUN mkdir -p /app/logs /app/data

   # 非 root 用户（推荐）
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser

   ENV APP_ID=my_app
   ENV DEBUG=0
   ENV LOG_LEVEL=INFO

   HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
     CMD python3 -c "import sys; sys.exit(0)"

   CMD ["python3", "app.py"]

应用清单（app.yaml）
~~~~~~~~~~~~~~~~~~~~

最小可用的 ``app.yaml`` 如下（字段详解见 `app_yaml_reference`_）：

.. code-block:: yaml

   apiVersion: v1
   kind: Application

   metadata:
     id: my_app
     name: My Application
     version: 1.0.0
     description: Person detection demo

   spec:
     image: my-app:1.0.0
     permissions:
       video:
         - cam0_main.raw
       inference:
         models: ["person_v1"]

.. note::

   官方应用仓库 `neoruntime-apps <https://github.com/camthink-ai/neoruntime-apps>`_
   的 ``templates/basic/`` 提供完整 manifest 模板，各 example/showcase 目录
   也可作为参考。

.. _app_image_step2:

步骤 2: 构建 Docker 镜像
-------------------------

在应用目录下执行构建命令：

.. code-block:: bash

   docker build -t my-app:1.0.0 .

构建参数说明：

- ``-t my-app:1.0.0`` — 镜像名称和标签，需与 ``app.yaml`` 中的 ``spec.image`` 一致
- ``.`` — 构建上下文为当前目录

验证镜像构建成功：

.. code-block:: bash

   docker images | grep my-app

.. note::

   如果应用需要额外依赖，可在目录中添加 ``requirements.txt`` 并在 Dockerfile 中加入
   ``RUN pip install -r requirements.txt``。如果需要离线或可重复构建，建议先构建 SDK
   wheel，将 ``neoruntime_ipc_sdk-*.whl`` 复制进镜像，再安装这个本地 wheel；亦可从
   源码安装并钉住发布 tag：
   ``git+https://github.com/camthink-ai/neoruntime-sdks.git@v0.7.4#subdirectory=python``。

.. _app_image_step3:

步骤 3: 导出镜像并打包 .neoapp
-------------------------------

将构建好的镜像导出为 tar 文件（打包时统一命名为 ``image.tar``）：

.. code-block:: bash

   docker save my-app:1.0.0 -o image.tar

然后将 ``app.yaml`` 与 ``image.tar`` 组装为 ``.neoapp`` 应用包：

.. code-block:: bash

   PKG=my-app-1.0.0-arm64
   mkdir -p "$PKG"
   cp app.yaml "$PKG/"
   mv image.tar "$PKG/"
   (cd "$PKG" && sha256sum app.yaml image.tar > SHA256SUMS)
   tar -czf "$PKG.neoapp" "$PKG"

``.neoapp`` 本质是 tar.gz：包内需含 ``app.yaml`` 与 ``image.tar``（放在根下
或唯一子目录中均可），``SHA256SUMS`` 等其余文件会被安装端忽略。gzip 同时承担
传输压缩，无需再单独压缩。

.. note::

   官方应用仓库 `neoruntime-apps <https://github.com/camthink-ai/neoruntime-apps>`_
   的 ``scripts/build_app.sh <应用目录>`` 一键完成 docker build → save →
   ``.neoapp`` 打包；其 Releases 还提供预构建的 showcase 应用包
   （``*-arm64.neoapp``）可直接下载导入。

.. _app_image_step4:

步骤 4: Web Console 导入（推荐）
---------------------------------

Web Console 的 **导入应用** 对话框共三屏：选择来源 → 配置应用 → 安装进度。

.. note::

   单一上传槽支持的文件：``.neoapp`` 应用包（推荐，服务端自动解出
   ``app.yaml`` 与镜像），或裸镜像 ``.tar`` / ``.tar.gz`` / ``.tgz``，
   最大 2GB。

打开导入对话框
~~~~~~~~~~~~~~

#. 打开浏览器，访问设备 Web Console：``http://<device-ip>:8080``
#. 导航到 **应用管理** 页面
#. 点击 **导入应用** 卡片，打开导入对话框

第 1 屏 — 选择来源
~~~~~~~~~~~~~~~~~~

- **本地上传** （默认，离线设备推荐）：拖入或选择一个 ``.neoapp`` 应用包
  （或裸镜像 tar），上传过程显示进度条
- **镜像仓库**：输入 Docker 镜像地址（如 ``docker.io/library/nginx:latest``），
  设备需可访问网络

上传 ``.neoapp`` 后，服务端自动解出包内 ``app.yaml`` 与镜像，后续表单以包内
``app.yaml`` 为准，可在下一屏微调；仅上传裸镜像时，配置完全由表单生成。

第 2 屏 — 配置应用
~~~~~~~~~~~~~~~~~~

单页表单按分区组织（侧栏分区导航，支持 **表单/YAML** 双视图切换）：

- **基本信息**：应用 ID、名称、版本、描述
- **资源**：CPU / 内存限制、共享内存（零拷贝视频流需要）、开机自启、重启策略
- **模型**：设备上可用的推理模型与最大 QPS
- **权限**：视频流、事件主题（支持通配符）、网络模式、设备控制
- **高级** （可选）：环境变量、卷挂载

第 3 屏 — 安装进度
~~~~~~~~~~~~~~~~~~

提交后显示安装任务进度；安装完成后应用出现在应用列表中。

.. _app_image_step5:

步骤 5: 命令行导入（备选方案）
-------------------------------

如果无法使用 Web Console，可先将 ``.neoapp`` 应用包通过 SCP 传输到设备，
解包后用 aipc-cli 安装。

传输应用包到设备：

.. code-block:: bash

   scp my-app-1.0.0-arm64.neoapp root@<device-ip>:/tmp/

然后 SSH 登录设备：

.. code-block:: bash

   # 解包 .neoapp（得到 app.yaml 与 image.tar）
   tar xzf /tmp/my-app-1.0.0-arm64.neoapp -C /tmp/

   # 安装应用（位置参数：manifest 在前，镜像 tar 在后）
   aipc-cli app install /tmp/my-app-1.0.0-arm64/app.yaml \
                        /tmp/my-app-1.0.0-arm64/image.tar

   # 启动应用
   aipc-cli app start my_app

   # 查看应用状态
   aipc-cli app list

   # 查看应用日志
   aipc-cli app logs my_app

也可以使用 gRPC 客户端直接调用 app-manager 服务：

.. code-block:: bash

   grpcurl -plaintext \
     -d '{"manifest_path": "/tmp/my-app-1.0.0-arm64/app.yaml",
          "image_path": "/tmp/my-app-1.0.0-arm64/image.tar"}' \
     unix:///run/aipc/app-manager.sock \
     appmanager.AppManager/InstallApp

.. _app_yaml_reference:

应用清单参考（app.yaml）
------------------------

以下是 ``app.yaml`` 各字段的详细说明。

元数据（metadata）
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - 字段
     - 必填
     - 说明
   * - id
     - 是
     - 应用唯一标识符（小写字母、数字、下划线）
   * - name
     - 是
     - 应用显示名称
   * - version
     - 是
     - 语义化版本号（如 1.0.0）
   * - description
     - 是
     - 应用描述
   * - author
     - 否
     - 作者名称
   * - email
     - 否
     - 联系邮箱

资源限制（spec.resources）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - 字段
     - 默认值
     - 说明
   * - cpu
     - —
     - CPU 限制，如 ``"50%"`` 或 ``"0.5"``
   * - memory
     - —
     - 内存限制，如 ``"256Mi"`` 或 ``"1Gi"``
   * - shm
     - false
     - 是否启用共享内存（零拷贝视频流需要）

权限配置（spec.permissions）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**视频流权限（video）**

指定应用可访问的视频流：

- ``cam0_main.raw`` — 原始视频流（通过 SHM 零拷贝）
- ``cam0_sub.raw`` — 子码流原始视频
- ``cam0_main`` — 编码视频流（通过 Unix socket）

.. note::

   权限层与 SDK 客户端的流命名不同层：manifest 权限按平台侧命名（如
   ``cam0_main.raw``）申请；SDK 调用（``FdMediaClient.subscribe`` /
   ``InferenceClient.subscribe``）按设备实际暴露的流 ID ``main`` / ``sub``
   订阅。

**AI 推理权限（inference）**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 字段
     - 默认值
     - 说明
   * - models
     - []
     - 可使用的模型列表
   * - max_qps
     - —
     - 最大 QPS 限制
   * - max_concurrent
     - —
     - 最大并发推理数

**事件总线权限（events）**

- ``publish`` — 可发布的事件主题（支持通配符 ``*``）
- ``subscribe`` — 可订阅的事件主题（支持通配符 ``*``）

**设备控制权限（device）**

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - 字段
     - 默认值
     - 说明
   * - light
     - false
     - 补光灯控制
   * - ir_cut
     - false
     - 红外滤光片控制
   * - ptz
     - false
     - 云台控制
   * - lens
     - false
     - 镜头变焦/对焦控制

**网络权限（network）**

- ``mode`` — 网络模式：``"isolated"``（默认）或 ``"host"``
- ``outbound`` — 允许的出站地址（isolated 模式下）

生命周期配置
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 字段
     - 默认值
     - 说明
   * - autostart
     - false
     - 系统启动时自动启动
   * - restart_policy
     - "no"
     - 重启策略：always / on-failure / no
   * - restart_max_retries
     - 3
     - 最大重启次数（on-failure 时）

健康检查（healthcheck）
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - 字段
     - 默认值
     - 说明
   * - enabled
     - false
     - 启用健康检查
   * - interval
     - 30s
     - 检查间隔
   * - timeout
     - 5s
     - 超时时间
   * - retries
     - 3
     - 失败重试次数

.. _app_image_faq:

常见问题
--------

镜像构建失败
~~~~~~~~~~~~

**错误**: ``failed to solve: failed to fetch``

检查网络连接，如需代理：

.. code-block:: bash

   docker build --build-arg HTTP_PROXY=http://proxy:port \
                --build-arg HTTPS_PROXY=http://proxy:port \
                -t my-app:1.0.0 .

应用包过大
~~~~~~~~~~

``.neoapp`` 本身已是 gzip 压缩。若仍然过大，检查镜像是否带入了不必要的
层或缓存（Dockerfile 使用 ``--no-cache-dir`` 安装依赖、多阶段构建等）：

.. code-block:: bash

   # 查看各层体积，定位大层
   docker history my-app:1.0.0

导入失败
~~~~~~~~

**错误**: ``Failed to import image to containerd``

.. code-block:: bash

   # 检查 containerd 状态
   systemctl status containerd

   # 手动导入测试
   ctr -n aipc images import /tmp/my-app-1.0.0-arm64/image.tar

权限错误
~~~~~~~~

**错误**: ``Permission denied``

.. code-block:: bash

   chmod 644 /tmp/my-app-1.0.0-arm64/app.yaml /tmp/my-app-1.0.0-arm64/image.tar
