r"""ColossalAI 内部使用的 transformers 版本兼容垫片（**不对外导出**）。

背景
----
ColossalAI 把 transformers 钉在 ``==4.51.3``（``requirements/requirements.txt``）。
迁移到 v5 时，一批被引用的 transformers 内部符号发生了变化，分两类：

1. **换位置** —— 如 ``no_init_weights`` 从 ``transformers.modeling_utils``
   搬到 ``transformers.initialization``；
2. **被删除** —— 如 ``is_remote_url`` / ``download_url`` / ``is_safetensors_available``
   在 v5 的 transformers 安装树里均**已 0 处残留**（见 ``scripts/N11_symbol_relocation_probe.py``）。

本模块把这两类差异收敛到一处，避免版本分支散落到 ``lazy`` / ``shardformer`` /
``checkpoint_io`` 各处。

约定（沿用本仓库既有风格，不是新引入的模式）
------------------------------------------
* **优先特性检测，不优先版本号字符串**：换位置类的差异一律用 ``try/except ImportError``
  双路径导入，天然同时覆盖 v4 与 v5，不依赖 ``transformers.__version__`` 的解析。
  仓库既有先例：``shardformer/policies/qwen2.py`` 用
  ``hasattr(self.model.config, "num_key_value_heads")`` 做特性检测。
* **能力被删除时不静默改语义**：返回 ``None`` 让调用点显式处理，由调用点抛出
  **带解释的报错**，而不是悄悄退化成另一种行为。
* **import 本模块无副作用**：不在模块顶层导入 transformers 的重物、不打日志 ——
  与 ``shardformer/_utils.py`` 同一规矩，``get_*`` 一律函数式惰性取用。

用法
----
    from colossalai._compat import get_no_init_weights, is_safetensors_available

    with get_no_init_weights()():
        ...

    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)
"""

from importlib.util import find_spec
from inspect import signature
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

# `get_auth_kwarg_name` 的结果缓存（惰性求值，避免每次加载模型都做一次签名内省）
_AUTH_KWARG_NAME: Optional[str] = None


def get_no_init_weights() -> Callable[[], Any]:
    r"""取得 ``no_init_weights`` 上下文管理器工厂。

    transformers v5 把 ``no_init_weights`` 定义在 ``transformers.initialization``
    （v5.17.0 实测：``initialization.py:254``）；v4 在 ``transformers.modeling_utils``。

    Returns:
        Callable[[], Any]: 无参可调用对象，返回一个上下文管理器。
    """
    try:
        from transformers.initialization import no_init_weights as _fn
    except ImportError:
        from transformers.modeling_utils import no_init_weights as _fn
    return _fn


def is_safetensors_available() -> bool:
    r"""``safetensors`` 是否可用。

    v4 由 ``transformers.utils.is_safetensors_available`` 提供；v5 该函数已删除
    （safetensors 转为硬依赖），退化为 ``importlib`` 探测。

    注意：本仓库另有一份同名实现 ``colossalai/checkpoint_io/utils.py:62``，
    其函数体是 ``try: return True / except ImportError: return False``，
    恒为 ``True``（是早期模块级 import 的遗留）。此处不复用那一份，以免把
    「有没有 safetensors」的探测变成常量返回。

    Returns:
        bool: 可用为 ``True``。
    """
    try:
        from transformers.utils import is_safetensors_available as _fn
    except ImportError:
        return find_spec("safetensors") is not None
    return bool(_fn())


def is_remote_url(url_or_filename: str) -> bool:
    r"""判断给定字符串是否是裸 URL（``http`` / ``https``）。

    该符号在 transformers v5 已删除，此处按 v4 语义就地实现（v4 的实现同为
    ``urlparse(...).scheme in ("http", "https")``）。保留它的用途是**给出解释性报错**，
    详见 :func:`get_download_url`。

    Args:
        url_or_filename (str): 待判断的路径或 URL。

    Returns:
        bool: 是裸 URL 为 ``True``。
    """
    return urlparse(str(url_or_filename)).scheme in ("http", "https")


def get_download_url() -> Optional[Callable[..., str]]:
    r"""取得「从裸 URL 下载权重」的实现。

    v4 返回 ``transformers.modeling_utils.download_url``；**v5 已删除该能力**——
    ``download_url`` 与 ``is_remote_url`` 在 v5 安装树里 0 处残留，
    ``transformers.utils.hub.cached_file`` 的签名也收缩为
    ``(path_or_repo_id, filename, **kwargs)``，不再接受 URL。

    因此 v5 下返回 ``None``。调用点必须显式处理 ``None``（抛带解释的报错），
    不要静默落到「把它当 repo id 去缓存里找」的分支 —— 那样用户拿到的是
    一个看不出真实原因的 ``OSError``。

    Returns:
        Optional[Callable[..., str]]: v4 为可调用对象，v5 为 ``None``。
    """
    try:
        from transformers.modeling_utils import download_url as _fn
    except ImportError:
        return None
    return _fn


def get_auth_kwarg_name() -> str:
    r"""取得当前 transformers 版本使用的**认证参数名**。

    v5 把 ``use_auth_token`` 改名成 ``token``。实测（v5.17.0，4/4 入口）：

    ==========================================  ================  =======
    入口                                        use_auth_token    token
    ==========================================  ================  =======
    ``PretrainedConfig.from_pretrained``        无                有
    ``GenerationConfig.from_pretrained``        无                有
    ``transformers.utils.hub.has_file``         无                有
    ``transformers.utils.hub.cached_file``      无                有
    ==========================================  ================  =======

    ``cached_file`` 自身是 ``(path_or_repo_id, filename, **kwargs)``，不接受具名参数，
    但会把 kwargs 透传给 ``cached_files``，后者有 ``token``，故同样适用。

    为什么必须处理：``lazy/pretrained.py`` 把该参数显式传给
    ``PretrainedConfig.from_pretrained(..., return_unused_kwargs=True)``。v4 里它被识别，
    不进 unused；v5 里它成了「未识别参数」被原样退回 ``model_kwargs``，最终撞在模型
    构造函数上（``TypeError: ... unexpected keyword argument 'use_auth_token'``）。
    **用户什么都不传也会崩**（退回的是 ``None``），是必经路径而非边界情况。

    检测顺序：**先探 ``token``**。理由是两个名字都存在的过渡版本里，``token`` 是新名，
    取它不会踩到 ``use_auth_token`` 的弃用告警。

    Returns:
        str: ``"token"`` 或 ``"use_auth_token"``。取不到签名时兜底为新名 ``"token"``。
    """
    global _AUTH_KWARG_NAME
    if _AUTH_KWARG_NAME is None:
        try:
            from transformers.configuration_utils import PretrainedConfig as _Config

            _params = signature(_Config.from_pretrained).parameters
            _AUTH_KWARG_NAME = "token" if "token" in _params else "use_auth_token"
        except Exception:
            _AUTH_KWARG_NAME = "token"
    return _AUTH_KWARG_NAME


def auth_kwargs(token: Any = None) -> Dict[str, Any]:
    r"""构造「认证参数」kwargs，名字随 transformers 版本自动切换。

    调用点写法（把原来的 ``use_auth_token=use_auth_token`` 换掉）::

        config, model_kwargs = cls.config_class.from_pretrained(
            config_path, ..., **auth_kwargs(use_auth_token)
        )

    Args:
        token (Any): 认证 token。``None`` 表示不认证（也要显式传，见
            :func:`get_auth_kwarg_name` 的说明）。

    Returns:
        Dict[str, Any]: 形如 ``{"token": token}`` 或 ``{"use_auth_token": token}``。
    """
    return {get_auth_kwarg_name(): token}
