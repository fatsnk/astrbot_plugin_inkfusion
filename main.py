import os
import re
import ssl
import uuid
import random
import json
import asyncio
import tempfile
import base64
import urllib.parse
import aiohttp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp


@register(
    "astrbot_plugin_inkfusion",
    "F5",
    "使用 Pollinations AI生成图片，支持多API Key、模型切换、LLM提示词优化；可以接入多模态llm提供商,解析llm api返回的图片链接。也可以接stable diffusion（A1111）。",
    "0.1.0",
    "https://github.com/fatsnk/astrbot_plugin_inkfusion"
)
class InkfusionPlugin(Star):
    """
    通过 Pollinations AI 服务生成图片的插件。
    支持多 API Key 随机选用、多模型管理、可选 LLM 提示词优化。
    """

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config

        # API Keys
        self.api_keys: list = self.config.get("api_keys", [])

        # 模型列表，第一个为默认
        self.models: list = self.config.get("models", ["flux"])
        if not self.models:
            self.models = ["flux"]

        # 图片参数
        self.width: int = self.config.get("width", 1024)
        self.height: int = self.config.get("height", 1024)
        self.seed: int = self.config.get("seed", -1)
        self.enhance: bool = self.config.get("enhance", False)
        self.negative_prompt: str = self.config.get("negative_prompt", "worst quality, blurry")
        self.safe: bool = self.config.get("safe", False)
        self.quality: str = self.config.get("quality", "medium")

        # 提示词优化开关
        self.enable_prompt_optimization: bool = self.config.get("enable_prompt_optimization", True)

        # 提示词优化 provider
        self.prompt_provider_name: str = self.config.get("prompt_provider_name", "")

        # 优化系统提示词
        self.optimization_system_prompt: str = self.config.get(
            "optimization_system_prompt",
            "You are an expert in crafting prompts for AI image generation models. "
            "Your task is to take a user's simple idea and transform it into a rich, detailed, and artistic prompt in English. "
            "The final output should be a single, continuous string of keywords and descriptions, separated by commas. "
            "Do not add any other explanatory text, just the prompt itself. "
            "Focus on visual details, art style (e.g., photorealistic, watercolor, anime), composition, and lighting."
        )

        # LLM 直接生图配置
        self.llm_image_provider_name: str = self.config.get("llm_image_provider_name", "")
        self.llm_image_system_prompt: str = self.config.get(
            "llm_image_system_prompt",
            "你是一个图片生成助手。请根据用户的描述生成图片。直接生成图片，不要添加多余的解释文字。"
        )

        # 临时图片存储目录
        self.temp_dir = os.path.join(tempfile.gettempdir(), "pollinations_images")
        os.makedirs(self.temp_dir, exist_ok=True)

        # 请求配置
        self.max_retries = 3
        self.request_timeout = 300

        # Stable Diffusion (A1111) 配置
        self.sd_enabled: bool = self.config.get("sd_enabled", False)
        self.sd_skip_ssl_verify: bool = self.config.get("sd_skip_ssl_verify", False)
        self.sd_base_url: str = self.config.get("sd_base_url", "http://127.0.0.1:7860").rstrip("/")
        self.sd_width: int = self.config.get("sd_width", 512)
        self.sd_height: int = self.config.get("sd_height", 512)
        self.sd_positive_prompt: str = self.config.get("sd_positive_prompt", "masterpiece, best quality, {{positive}}")
        self.sd_negative_prompt: str = self.config.get("sd_negative_prompt", "bad quality, worst quality, low quality, blurry, bad anatomy, bad hands, extra digits")
        self.sd_steps: int = self.config.get("sd_steps", 20)
        self.sd_cfg_scale: float = float(self.config.get("sd_cfg_scale", 7.0))
        self.sd_sampler_name: str = self.config.get("sd_sampler_name", "Euler a")
        self.sd_scheduler: str = self.config.get("sd_scheduler", "")
        self.sd_seed: int = self.config.get("sd_seed", -1)
        self.sd_restore_faces: bool = self.config.get("sd_restore_faces", False)
        self.sd_model_checkpoint: str = self.config.get("sd_model_checkpoint", "")
        self.sd_clip_skip: int = self.config.get("sd_clip_skip", 0)

        logger.info(
            f"花粉AI图片生成插件已加载 | 模型: {self.models} | Keys: {len(self.api_keys)}个 | "
            f"优化: {'开' if self.enable_prompt_optimization else '关'} | "
            f"尺寸: {self.width}x{self.height} | "
            f"SD: {'开' if self.sd_enabled else '关'}"
        )

    def _extract_full_args(self, event: AstrMessageEvent, *prefixes: str) -> str:
        """从原始消息中提取命令后的完整参数文本。
        尝试多个前缀匹配，返回去掉前缀后的完整文本。
        """
        raw = event.message_str.strip()
        for prefix in prefixes:
            if raw.startswith(prefix):
                return raw[len(prefix):].strip()
        return raw

    def _get_random_api_key(self) -> str:
        """随机获取一个 API Key，无可用 key 则返回空字符串"""
        keys = [k for k in self.api_keys if k and k.strip()]
        return random.choice(keys).strip() if keys else ""

    def _get_current_model(self) -> str:
        """获取当前使用的模型（列表第一个）"""
        if self.models and self.models[0]:
            return self.models[0].strip()
        return "flux"

    def _build_query_params(self, model: str) -> str:
        """构建 URL 查询参数"""
        params = {
            "model": model,
            "width": self.width,
            "height": self.height,
            "nologo": "true",
        }

        if self.seed >= 0:
            params["seed"] = self.seed

        if self.enhance:
            params["enhance"] = "true"

        if self.negative_prompt:
            params["negative_prompt"] = self.negative_prompt

        if self.safe:
            params["safe"] = "true"

        if self.quality and self.quality != "medium":
            params["quality"] = self.quality

        return urllib.parse.urlencode(params)

    async def _optimize_prompt(self, theme: str) -> str:
        """通过 LLM 优化提示词"""
        provider = None
        if self.prompt_provider_name:
            provider = self.context.get_provider_by_id(self.prompt_provider_name)

        if not provider:
            provider = self.context.get_using_provider()

        if not provider:
            logger.warning("未找到可用的LLM服务，将使用原始提示词。")
            return theme

        llm_response = await provider.text_chat(
            prompt=f"User's idea: {theme}",
            system_prompt=self.optimization_system_prompt,
            contexts=[]
        )

        if not llm_response or not llm_response.completion_text:
            logger.error("LLM未能返回有效的提示词，使用原始输入。")
            return theme

        return llm_response.completion_text.strip()

    async def _generate_image(self, prompt_text: str, model: str = None) -> str:
        """
        生成图片并下载到本地临时文件，返回本地文件路径。
        支持重试机制。
        API: https://gen.pollinations.ai/image/{prompt}?model={model}&...
        认证: Authorization: Bearer {api_key}
        """
        if model is None:
            model = self._get_current_model()

        # 是否优化提示词
        if self.enable_prompt_optimization:
            refined_prompt = await self._optimize_prompt(prompt_text)
            logger.info(f"优化后提示词: {refined_prompt[:100]}...")
        else:
            refined_prompt = prompt_text

        # 对提示词进行完整 URL 编码，safe='' 确保空格等所有特殊字符都被编码
        encoded_prompt = urllib.parse.quote(refined_prompt, safe='')
        query_string = self._build_query_params(model)
        image_url = f"https://gen.pollinations.ai/image/{encoded_prompt}?{query_string}"

        logger.info(f"完整请求URL: {image_url}")

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            api_key = self._get_random_api_key()
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                logger.debug(f"尝试 {attempt}/{self.max_retries} | API Key: {api_key[:8]}...")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        image_url,
                        headers=headers,
                        allow_redirects=True,
                        timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                    ) as resp:
                        if resp.status == 200:
                            # 检查内容类型是否为图片
                            content_type = resp.headers.get("Content-Type", "")
                            if "image" not in content_type:
                                error_text = await resp.text()
                                logger.warning(f"尝试 {attempt}: 响应非图片类型 ({content_type}): {error_text[:100]}")
                                last_error = f"响应非图片类型: {content_type}"
                                continue

                            # 下载图片到本地临时文件
                            image_data = await resp.read()
                            if not image_data:
                                logger.warning(f"尝试 {attempt}: 图片数据为空")
                                last_error = "图片数据为空"
                                continue

                            # 根据 content-type 确定扩展名
                            ext = ".jpg"
                            if "png" in content_type:
                                ext = ".png"
                            elif "webp" in content_type:
                                ext = ".webp"

                            filename = f"{uuid.uuid4().hex}{ext}"
                            filepath = os.path.join(self.temp_dir, filename)

                            with open(filepath, "wb") as f:
                                f.write(image_data)

                            logger.info(f"图片下载成功: {filepath} ({len(image_data)} bytes)")
                            return filepath
                        else:
                            error_text = await resp.text()
                            logger.warning(f"尝试 {attempt}: HTTP {resp.status}: {error_text[:200]}")
                            last_error = f"HTTP {resp.status}"

            except asyncio.TimeoutError:
                logger.warning(f"尝试 {attempt}/{self.max_retries}: 请求超时 ({self.request_timeout}s)")
                last_error = f"请求超时 ({self.request_timeout}s)"
            except aiohttp.ClientError as e:
                logger.warning(f"尝试 {attempt}/{self.max_retries}: 网络错误: {e}")
                last_error = f"网络错误: {str(e)}"

            # 重试前等待
            if attempt < self.max_retries:
                wait_time = attempt * 2
                logger.debug(f"等待 {wait_time}s 后重试...")
                await asyncio.sleep(wait_time)

        raise Exception(f"图片生成失败，已重试{self.max_retries}次。最后错误: {last_error}")

    def _extract_media_urls(self, text: str) -> dict:
        """从文本中提取图片和视频 URL。
        返回 {"images": [...], "videos": [...]}
        """
        images = []
        videos = []

        # Markdown 图片: ![alt](url)
        md_imgs = re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text)
        images.extend(md_imgs)

        # HTML img src
        html_imgs = re.findall(r'<img[^>]+src=["\']?(https?://[^\s"\'>\)]+)', text)
        images.extend(html_imgs)

        # HTML video/source src
        html_videos = re.findall(r'<source[^>]+src=["\']?(https?://[^\s"\'>\)]+)', text)
        videos.extend(html_videos)
        html_video_tags = re.findall(r'<video[^>]+src=["\']?(https?://[^\s"\'>\)]+)', text)
        videos.extend(html_video_tags)

        # 裸 URL 匹配（未被上面捕获的）
        all_urls = re.findall(r'(https?://[^\s\)\]"\'<>]+)', text)
        img_exts = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
        vid_exts = ('.mp4', '.webm', '.mov', '.avi')

        for url in all_urls:
            # 去掉 URL 中可能的查询参数来判断扩展名
            path_part = urllib.parse.urlparse(url).path.lower()
            if any(path_part.endswith(ext) for ext in img_exts):
                if url not in images:
                    images.append(url)
            elif any(path_part.endswith(ext) for ext in vid_exts):
                if url not in videos:
                    videos.append(url)

        return {"images": images, "videos": videos}

    async def _download_media(self, url: str) -> str:
        """下载媒体文件到本地临时目录，返回本地文件路径。"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    allow_redirects=True,
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"下载媒体失败 HTTP {resp.status}: {url}")
                        return None

                    data = await resp.read()
                    if not data:
                        logger.error(f"下载媒体数据为空: {url}")
                        return None

                    # 从 URL 路径或 content-type 推断扩展名
                    content_type = resp.headers.get("Content-Type", "")
                    path_part = urllib.parse.urlparse(url).path.lower()

                    ext = ".bin"
                    if "image/png" in content_type or path_part.endswith(".png"):
                        ext = ".png"
                    elif "image/webp" in content_type or path_part.endswith(".webp"):
                        ext = ".webp"
                    elif "image/gif" in content_type or path_part.endswith(".gif"):
                        ext = ".gif"
                    elif "image" in content_type or any(path_part.endswith(e) for e in ('.jpg', '.jpeg')):
                        ext = ".jpg"
                    elif "video/mp4" in content_type or path_part.endswith(".mp4"):
                        ext = ".mp4"
                    elif "video/webm" in content_type or path_part.endswith(".webm"):
                        ext = ".webm"
                    elif "video" in content_type:
                        ext = ".mp4"

                    filename = f"{uuid.uuid4().hex}{ext}"
                    filepath = os.path.join(self.temp_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(data)

                    logger.info(f"媒体下载成功: {filepath} ({len(data)} bytes)")
                    return filepath

        except Exception as e:
            logger.error(f"下载媒体异常: {url} | {e}")
            return None

    async def _generate_via_llm(self, prompt_text: str) -> dict:
        """通过 LLM 直接生成图片/视频。
        返回 {"images": [本地路径...], "videos": [本地路径...], "text": 原始文本}
        """
        if not self.llm_image_provider_name:
            raise Exception("未配置 LLM 生图提供商，请在插件设置中选择。")

        provider = self.context.get_provider_by_id(self.llm_image_provider_name)
        if not provider:
            raise Exception(f"未找到 LLM 提供商: {self.llm_image_provider_name}")

        llm_response = await provider.text_chat(
            prompt=prompt_text,
            system_prompt=self.llm_image_system_prompt,
            contexts=[]
        )

        if not llm_response or not llm_response.completion_text:
            raise Exception("LLM 未返回有效内容。")

        raw_text = llm_response.completion_text.strip()
        logger.info(f"LLM 生图原始返回 ({len(raw_text)} chars): {raw_text[:200]}...")

        # 提取媒体链接
        media = self._extract_media_urls(raw_text)
        logger.info(f"提取到媒体: {len(media['images'])} 张图片, {len(media['videos'])} 个视频")

        result = {"images": [], "videos": [], "text": raw_text}

        # 下载图片
        for img_url in media["images"]:
            local_path = await self._download_media(img_url)
            if local_path:
                result["images"].append(local_path)

        # 下载视频
        for vid_url in media["videos"]:
            local_path = await self._download_media(vid_url)
            if local_path:
                result["videos"].append(local_path)

        return result

    async def _generate_via_qwen_image(self, prompt_text: str) -> str:
        """通过千问图像生成 API 直接生图，下载到本地并返回文件路径。"""
        provider = self.context.get_provider_by_id(self.llm_image_provider_name)
        if not provider:
            raise Exception(f"未找到 LLM 提供商: {self.llm_image_provider_name}")

        # 从 provider ID 解析 source_id（格式: "Qwen/qwen-image-2.0" → "Qwen"）
        pid = getattr(provider, 'id', '') or self.llm_image_provider_name
        source_id = pid.split('/')[0] if '/' in pid else pid

        # 读取 AstrBot 主配置获取 provider source 的 key 和 api_base
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(data_dir, '..', 'cmd_config.json')
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            main_config = json.load(f)

        src_config = None
        for src in main_config.get('provider_sources', []):
            if src.get('id') == source_id:
                src_config = src
                break

        if not src_config:
            raise Exception(f"未在配置中找到 provider source: {source_id}")

        keys = src_config.get("key", [])
        if not keys or not keys[0]:
            raise Exception(f"Provider source '{source_id}' 未配置 API Key")

        api_key = keys[0]
        api_base = src_config.get("api_base", "https://dashscope.aliyuncs.com")

        # 从 api_base 提取域名，拼接多模态生图端点
        parsed = urllib.parse.urlparse(api_base)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        endpoint = f"{base_domain}/api/v1/services/aigc/multimodal-generation/generation"

        payload = {
            "model": "qwen-image-2.0-pro",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt_text}]
                    }
                ]
            },
            "parameters": {
                "n": 1,
                "negative_prompt": " ",
                "prompt_extend": True,
                "watermark": False,
                "size": f"{self.width}*{self.height}"
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        logger.info(f"千问生图请求: endpoint={endpoint} source={source_id} size={self.width}x{self.height}")

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"千问图像生成失败: HTTP {resp.status} - {error_text[:300]}")
                result_json = await resp.json()

        image_url = result_json["output"]["choices"][0]["message"]["content"][0]["image"]
        logger.info(f"千问返回图片URL: {image_url[:80]}...")

        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as resp:
                if resp.status != 200:
                    raise Exception(f"下载千问图片失败: HTTP {resp.status}")
                image_data = await resp.read()

        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)

        logger.info(f"千问图片已保存: {filepath} ({len(image_data)} bytes)")
        return filepath

    def _get_sd_connector(self) -> aiohttp.TCPConnector:
        """获取 SD 请求用的 TCP 连接器，根据配置决定是否跳过 SSL 验证"""
        if self.sd_skip_ssl_verify:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            return aiohttp.TCPConnector(ssl=ssl_ctx)
        return aiohttp.TCPConnector()

    def _build_sd_prompt(self, user_prompt: str) -> str:
        """将用户提示词填入 SD 正面提示词模板的 {{positive}} 占位符中"""
        template = self.sd_positive_prompt
        if "{{positive}}" in template:
            return template.replace("{{positive}}", user_prompt)
        # 模板中没有占位符，直接拼接
        return f"{template}, {user_prompt}" if template else user_prompt

    async def _generate_image_sd(self, prompt_text: str) -> str:
        """
        通过 Stable Diffusion (Automatic1111) txt2img API 生成图片。
        返回本地临时文件路径。
        """
        endpoint = f"{self.sd_base_url}/sdapi/v1/txt2img"

        # 构建正面提示词（模板替换）
        positive = self._build_sd_prompt(prompt_text)

        # 构建请求参数
        sd_params = {
            "prompt": positive,
            "negative_prompt": self.sd_negative_prompt,
            "steps": self.sd_steps,
            "cfg_scale": self.sd_cfg_scale,
            "width": self.sd_width,
            "height": self.sd_height,
            "sampler_name": self.sd_sampler_name,
            "seed": self.sd_seed,
            "restore_faces": self.sd_restore_faces,
        }

        if self.sd_scheduler:
            sd_params["scheduler"] = self.sd_scheduler

        # 构建 override_settings（模型、CLIP Skip 等）
        override_settings = {}
        if self.sd_model_checkpoint:
            override_settings["sd_model_checkpoint"] = self.sd_model_checkpoint
        if self.sd_clip_skip and self.sd_clip_skip > 0:
            override_settings["CLIP_stop_at_last_layers"] = self.sd_clip_skip

        if override_settings:
            sd_params["override_settings"] = override_settings
            sd_params["override_settings_restore_afterwards"] = True

        logger.info(f"SD 请求: {endpoint}")
        logger.debug(
            f"SD 参数: prompt={positive[:80]}..., negative={self.sd_negative_prompt[:40]}..., "
            f"steps={self.sd_steps}, cfg={self.sd_cfg_scale}, sampler={self.sd_sampler_name}, "
            f"size={self.sd_width}x{self.sd_height}"
        )

        try:
            connector = self._get_sd_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    endpoint,
                    json=sd_params,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300)  # SD 生图可能较慢
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"SD API 返回 HTTP {resp.status}: {error_text[:300]}")

                    data = await resp.json()

                    images = data.get("images", [])
                    if not images:
                        raise Exception("SD API 返回数据中没有图片")

                    # 第一张图片是 base64 编码的 PNG
                    image_bytes = base64.b64decode(images[0])

                    filename = f"{uuid.uuid4().hex}.png"
                    filepath = os.path.join(self.temp_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    logger.info(f"SD 图片生成成功: {filepath} ({len(image_bytes)} bytes)")
                    return filepath

        except aiohttp.ClientError as e:
            raise Exception(f"SD API 连接失败: {e}（请检查 {self.sd_base_url} 是否可访问）")
        except asyncio.TimeoutError:
            raise Exception("SD API 请求超时（生图可能需要较长时间，请检查 WebUI 状态）")

    async def _fetch_model_list(self) -> list:
        """从 Pollinations API 获取可用模型列表"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://gen.pollinations.ai/image/models",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.error(f"获取模型列表失败 HTTP {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"获取模型列表异常: {e}")
            return []

    @filter.llm_tool(name="generate_image_with_theme")
    async def generate_image_tool(self, event: AstrMessageEvent, theme: str):
        """
        LLM 函数调用工具：根据主题生成图片。

        Args:
            theme(string): 图片的详细描述
        """
        try:
            # 如果配置了 LLM 生图提供商，优先走千问图像生成 API
            if self.llm_image_provider_name:
                return await self._generate_via_qwen_image(theme)
            # 否则走 Pollinations API 通道
            return await self._generate_image(theme)
        except Exception as e:
            logger.error(f"图片生成过程中发生错误: {e}")
            raise Exception(f"生成图片时遇到问题: {str(e)}")

    @filter.command_group("ai生图")
    def image_cmd_group(self):
        """Pollinations AI 图片生成指令组"""
        pass

    @image_cmd_group.command("模型列表")
    async def list_models(self, event: AstrMessageEvent):
        """查看可用的图片生成模型列表。用法: /ai生图 模型列表"""
        yield event.plain_result("正在获取模型列表...")
        models = await self._fetch_model_list()
        if not models:
            yield event.plain_result("获取模型列表失败，请稍后再试。")
            return

        lines = [f"📋 可用模型列表 (共{len(models)}个):\n"]
        for m in models:
            name = m.get("name", "unknown")
            desc = m.get("description", "")
            paid = "💰" if m.get("paid_only", False) else "🆓"
            marker = " 👈 当前" if name == self._get_current_model() else ""
            lines.append(f"  {paid} {name} - {desc}{marker}")

        lines.append(f"\n当前配置模型: {', '.join(self.models)}")
        lines.append("切换模型: /ai生图 模型 <模型名称>")
        yield event.plain_result("\n".join(lines))

    @image_cmd_group.command("模型")
    async def switch_model(self, event: AstrMessageEvent, model_name: str):
        """切换默认图片生成模型。用法: /ai生图 模型 [名称]"""
        # 从原始消息提取完整模型名（模型名通常无空格，但保险起见）
        full_model_name = self._extract_full_args(event, "/ai生图 模型 ", "/ai生图 模型", "ai生图 模型 ", "ai生图 模型")
        if full_model_name:
            model_name = full_model_name

        if not model_name:
            yield event.plain_result("请指定模型名称，例如: /ai生图 模型 flux")
            return

        # 将指定模型放到列表首位
        if model_name in self.models:
            self.models.remove(model_name)
        self.models.insert(0, model_name)
        self.config["models"] = self.models

        yield event.plain_result(
            f"✅ 已切换默认模型为: {model_name}\n当前模型列表: {', '.join(self.models)}"
        )

    @image_cmd_group.command("生成")
    async def generate_image_sub(self, event: AstrMessageEvent, prompt_text: str):
        """根据描述生成图片。用法: /ai生图 生成 [描述]"""
        # 从原始消息提取完整提示词
        full_prompt = self._extract_full_args(event, "/ai生图 生成 ", "/ai生图 生成", "ai生图 生成 ", "ai生图 生成")
        if full_prompt:
            prompt_text = full_prompt

        if not prompt_text:
            yield event.plain_result("请输入图片描述，例如: /ai生图 生成 一只猫在太空漫步")
            return

        try:
            yield event.plain_result(
                f"🎨 正在生成图片，模型: {self._get_current_model()}，"
                f"尺寸: {self.width}x{self.height}，请稍候..."
            )
            image_path = await self._generate_image(prompt_text)
            yield event.image_result(image_path)
        except Exception as e:
            logger.error(f"图片生成过程中发生错误: {e}")
            yield event.plain_result(f"生成图片时遇到问题: {str(e)}")

    @image_cmd_group.command("llm")
    async def generate_via_llm_cmd(self, event: AstrMessageEvent, prompt_text: str):
        """使用LLM直接生成图片/视频。用法: /ai生图 llm [描述]"""
        # 从原始消息提取完整提示词
        full_prompt = self._extract_full_args(event, "/ai生图 llm ", "/ai生图 llm", "ai生图 llm ", "ai生图 llm")
        if full_prompt:
            prompt_text = full_prompt

        if not prompt_text:
            yield event.plain_result("请输入描述，例如: /ai生图 llm 一只猫在太空漫步")
            return

        if not self.llm_image_provider_name:
            yield event.plain_result("❌ 未配置 LLM 生图提供商，请在插件设置中选择。")
            return

        try:
            yield event.plain_result(f"🤖 正在通过 LLM 生成，请稍候...")

            result = await self._generate_via_llm(prompt_text)

            # 构建消息链
            chain = []

            # 添加图片
            for img_path in result["images"]:
                chain.append(Comp.Image.fromFileSystem(img_path))

            # 添加视频
            for vid_path in result["videos"]:
                chain.append(Comp.Video.fromFileSystem(path=vid_path))

            if chain:
                yield event.chain_result(chain)
            else:
                # 没有提取到媒体，返回 LLM 原始文本
                yield event.plain_result(f"LLM 返回内容（未检测到图片/视频）:\n{result['text'][:500]}")

        except Exception as e:
            logger.error(f"LLM 生图失败: {e}")
            yield event.plain_result(f"LLM 生图失败: {str(e)}")

    @filter.command("画")
    async def generate_image_shortcut(self, event: AstrMessageEvent, prompt_text: str):
        """快捷生图指令。用法: /画 [描述]"""
        # 从原始消息提取完整提示词
        full_prompt = self._extract_full_args(event, "/画 ", "/画", "画 ", "画")
        if full_prompt:
            prompt_text = full_prompt

        if not prompt_text:
            yield event.plain_result(
                "使用方法:\n"
                "  /画 [描述] - 快捷生成图片\n"
                "  /ai生图 生成 [描述] - 生成图片\n"
                "  /ai生图 模型列表 - 查看可用模型\n"
                "  /ai生图 模型 [名称] - 切换默认模型"
            )
            return

        try:
            yield event.plain_result(
                f"🎨 正在生成图片，模型: {self._get_current_model()}，"
                f"尺寸: {self.width}x{self.height}，请稍候..."
            )
            image_path = await self._generate_image(prompt_text)
            yield event.image_result(image_path)
        except Exception as e:
            logger.error(f"图片生成过程中发生错误: {e}")
            yield event.plain_result(f"生成图片时遇到问题: {str(e)}")

    # ==================== Stable Diffusion (A1111) 指令 ====================

    @filter.command_group("sd生图")
    def sd_cmd_group(self):
        """Stable Diffusion (A1111) 图片生成指令组"""
        pass

    @sd_cmd_group.command("生成")
    async def sd_generate(self, event: AstrMessageEvent, prompt_text: str):
        """使用 SD 生成图片。用法: /sd生图 生成 [描述]"""
        if not self.sd_enabled:
            yield event.plain_result("❌ Stable Diffusion 生图未启用，请在插件设置中开启。")
            return

        full_prompt = self._extract_full_args(event, "/sd生图 生成 ", "/sd生图 生成", "sd生图 生成 ", "sd生图 生成")
        if full_prompt:
            prompt_text = full_prompt

        if not prompt_text:
            yield event.plain_result("请输入图片描述，例如: /sd生图 生成 一个女孩在花园里")
            return

        # 可选：使用 LLM 优化提示词
        if self.enable_prompt_optimization:
            try:
                prompt_text = await self._optimize_prompt(prompt_text)
                logger.info(f"SD 优化后提示词: {prompt_text[:100]}...")
            except Exception as e:
                logger.warning(f"提示词优化失败，使用原始输入: {e}")

        try:
            final_positive = self._build_sd_prompt(prompt_text)
            yield event.plain_result(
                f"🎨 SD 生图中...\n"
                f"采样器: {self.sd_sampler_name} | 步数: {self.sd_steps} | CFG: {self.sd_cfg_scale}\n"
                f"尺寸: {self.sd_width}x{self.sd_height}\n"
                f"正面: {final_positive[:80]}..."
            )
            image_path = await self._generate_image_sd(prompt_text)
            yield event.image_result(image_path)
        except Exception as e:
            logger.error(f"SD 生图失败: {e}")
            yield event.plain_result(f"SD 生图失败: {str(e)}")

    @sd_cmd_group.command("采样器列表")
    async def sd_list_samplers(self, event: AstrMessageEvent):
        """查看 SD 可用采样器。用法: /sd生图 采样器列表"""
        if not self.sd_enabled:
            yield event.plain_result("❌ Stable Diffusion 未启用。")
            return

        try:
            connector = self._get_sd_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{self.sd_base_url}/sdapi/v1/samplers",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        yield event.plain_result(f"获取采样器列表失败 HTTP {resp.status}")
                        return
                    samplers = await resp.json()

            lines = [f"📋 SD 可用采样器 (共{len(samplers)}个):\n"]
            forin samplers:
                name = s.get("name", "unknown")
                marker = " 👈 当前" if name == self.sd_sampler_name else ""
                lines.append(f"  • {name}{marker}")
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {e}")

    @sd_cmd_group.command("模型列表")
    async def sd_list_models(self, event: AstrMessageEvent):
        """查看 SD 可用模型。用法: /sd生图 模型列表"""
        if not self.sd_enabled:
            yield event.plain_result("❌ Stable Diffusion 未启用。")
            return

        try:
            connector = self._get_sd_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{self.sd_base_url}/sdapi/v1/sd-models",
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        yield event.plain_result(f"获取模型列表失败 HTTP {resp.status}")
                        return
                    models = await resp.json()

            lines = [f"📋 SD 可用模型 (共{len(models)}个):\n"]
            for m in models:
                title = m.get("title", "unknown")
                model_name = m.get("model_name", "")
                marker = " 👈 当前" if model_name == self.sd_model_checkpoint or title == self.sd_model_checkpoint else ""
                lines.append(f"  • {title}{marker}")
            yield event.plain_result("\n".join(lines))
        except Exception as e:
            yield event.plain_result(f"获取模型列表失败: {e}")

    @filter.command("sd画")
    async def sd_generate_shortcut(self, event: AstrMessageEvent, prompt_text: str):
        """SD 快捷生图。用法: /sd画 [描述]"""
        if not self.sd_enabled:
            yield event.plain_result("❌ Stable Diffusion 未启用，请在插件设置中开启。")
            return

        full_prompt = self._extract_full_args(event, "/sd画 ", "/sd画", "sd画 ", "sd画")
        if full_prompt:
            prompt_text = full_prompt

        if not prompt_text:
            yield event.plain_result(
                "SD 生图用法:\n"
                "  /sd画 [描述] - 快捷生成\n"
                "  /sd生图 生成 [描述] - 生成图片\n"
                "  /sd生图 采样器列表 - 查看采样器\n"
                "  /sd生图 模型列表 - 查看模型"
            )
            return

        # 可选：使用 LLM 优化提示词
        if self.enable_prompt_optimization:
            try:
                prompt_text = await self._optimize_prompt(prompt_text)
            except Exception:
                pass

        try:
            final_positive = self._build_sd_prompt(prompt_text)
            yield event.plain_result(
                f"🎨 SD 生图中... | {self.sd_sampler_name} | {self.sd_steps}步 | {self.sd_width}x{self.sd_height}"
            )
            image_path = await self._generate_image_sd(prompt_text)
            yield event.image_result(image_path)
        except Exception as e:
            logger.error(f"SD 生图失败: {e}")
            yield event.plain_result(f"SD 生图失败: {str(e)}")

    # ==================== 通用工具方法 ====================

    def _cleanup_temp_files(self):
        """清理临时图片文件"""
        try:
            if os.path.exists(self.temp_dir):
                for f in os.listdir(self.temp_dir):
                    filepath = os.path.join(self.temp_dir, f)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                logger.debug("临时图片文件已清理。")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

    async def terminate(self):
        """插件卸载时调用"""
        self._cleanup_temp_files()
        logger.info("花粉AI图片生成插件已卸载。")
