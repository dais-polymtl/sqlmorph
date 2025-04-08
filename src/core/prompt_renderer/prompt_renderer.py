from typing import Dict, Any

import jinja2
from src.core.logger import Logger

logger = Logger(__name__)


class PromptRenderer:
    def __init__(self, templates_dir_path: str):
        self.templates_dir_path = templates_dir_path
        self.template_cache = {}

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        try:
            template = self._load_template(f"{template_name}.jinja")
            return template.render(context)
        except Exception as e:
            logger.log("error", "ERROR_IN_PROMPT_RENDERER", {"error": str(e)})
            raise

    def _load_template(self, template_file: str) -> jinja2.Template:
        if template_file not in self.template_cache:
            template_loader = jinja2.FileSystemLoader(
                searchpath=self.templates_dir_path
            )
            template_env = jinja2.Environment(loader=template_loader)
            self.template_cache[template_file] = template_env.get_template(
                template_file
            )
        return self.template_cache[template_file]
