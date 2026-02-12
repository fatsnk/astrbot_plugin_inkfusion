# astrbot_plugin_inkfusion
用于Astrbot的Pollinations、SD（A1111）、llm图片生成插件。
参考项目：https://github.com/qa296/astrbot_plugin_pollinations_images
；感谢这位大佬的工作。由于现在Pollinations的api已经更新，所以调整了功能，并添加更多生图方式，例如stable diffusion。

提醒：pollinations.ai接口已经更新。如果你使用pollinations进行文生图，请前往新的[管理页面](https://enter.pollinations.ai/)`https://enter.pollinations.ai/`创建新的api key。

1. 功能简介
Pollinations指令：

查看Pollinations可用的图片生成模型列表:

/ai生图 模型列表

切换默认Pollinations图片生成模型：

/ai生图 模型 模型名称

调用Pollinations的API生成图片：

/ai生图 生成 这里是提示词，例如画一只小猫

或者直接使用快捷指令`/画 XXX`

llm指令（只支持在回复文本中返回图片链接的api，例如markdown和html格式）：

使用llm直接生成图片/视频：

/ai生图 llm XXX...

stable diffusion指令：

/sd生图 生成 XXX...

或者使用快捷指令`/sd画 XXX`

- 更多指令请在Astrbot中的管理行为菜单查看。

2. 安装与启用
将插件文件夹放至 data/plugins。

确保 AstrBot 已启用「LLM 服务」：

astrbot dashboard → 服务管理 → 添加/启用一个文本模型（如 OpenAI、Ollama 等）。

重启 AstrBot，日志应出现：

花粉AI图片生成插件已加载。
