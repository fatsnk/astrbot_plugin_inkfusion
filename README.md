# astrbot_plugin_inkfusion

> 用于 AstrBot 的 Pollinations、SD（A1111）、LLM 图片生成插件

## 📌 项目说明

本插件为 AstrBot 提供多种 AI 图片生成方式，支持 Pollinations、Stable Diffusion 和 LLM 生图。

**参考项目：** [astrbot_plugin_pollinations_images](https://github.com/qa296/astrbot_plugin_pollinations_images)  
感谢原作者的工作！由于 Pollinations API 已更新，本插件进行了功能调整并添加了更多生图方式。

### ⚠️ 重要提醒

Pollinations.ai 接口已更新。如果使用 Pollinations 进行文生图，请前往新的 [管理页面](https://enter.pollinations.ai/) 创建新的 API Key。

---

## 🎨 功能简介

### Pollinations 指令

- **查看可用模型列表**
  ```
  /ai生图 模型列表
  ```

- **切换默认生成模型**
  ```
  /ai生图 模型 模型名称
  ```

- **生成图片**
  ```
  /ai生图 生成 这里是提示词，例如画一只小猫
  ```
  或使用快捷指令：
  ```
  /画 XXX
  ```

### LLM 指令

> 仅支持在回复文本中返回图片链接的 API（如 Markdown 和 HTML 格式）

- **使用 LLM 生成图片/视频**
  ```
  /ai生图 llm XXX...
  ```

### Stable Diffusion 指令

- **SD 生成图片**
  ```
  /sd生图 生成 XXX...
  ```
  或使用快捷指令：
  ```
  /sd画 XXX
  ```

💡 **更多指令**请在 AstrBot webUI界面的「管理行为」菜单中查看。

---

## 📦 安装与启用

### 1. 安装插件

将插件文件夹放置到 `data/plugins` 目录下。

### 2. 启用 LLM 服务

确保 AstrBot 已启用「LLM 服务」：

1. 进入 `astrbot dashboard`
2. 导航到 **服务管理**
3. 添加/启用一个文本模型（如 OpenAI、Ollama 等）

### 3. 重启 AstrBot

重启后，日志中应显示：

```
花粉AI图片生成插件已加载。
```

### 4. 配置插件

在插件配置中填写：
- Pollinations API Key
- 选择支持的 LLM 生图提供商
- 你的 SD URL 等配置信息

---

## 📝 许可与致谢

本项目基于原 [astrbot_plugin_pollinations_images](https://github.com/qa296/astrbot_plugin_pollinations_images) 项目开发，特此致谢！
