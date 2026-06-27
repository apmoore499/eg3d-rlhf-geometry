import re
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

# -----------------------------
from core_modules.utils import pylogger_c

# 0----------------------------------

log = pylogger_c.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "tree"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(f"Field '{field}' not found in config. Skipping '{field}' config printing...")

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # ['default', 'emacs', 'friendly', 'friendly_grayscale', 'colorful', 'autumn', 'murphy', 'manni', 'material', 'monokai',
    # 'perldoc', 'pastie', 'borland', 'trac', 'native', 'fruity', 'bw', 'vim', 'vs', 'tango',
    # 'rrt', 'xcode', 'igor', 'paraiso-light', 'paraiso-dark', 'lovelace', 'algol', 'algol_nu', 'arduino',
    # 'rainbow_dash', 'abap', 'solarized-dark', 'solarized-light', 'sas', 'staroffice', 'stata', 'stata-light',
    # 'stata-dark', 'inkpot', 'zenburn', 'gruvbox-dark', 'gruvbox-light', 'dracula', 'one-dark', 'lilypond', 'nord',
    # 'nord-darker', 'github-dark']

    from pygments.styles import get_all_styles

    styles = list(get_all_styles())

    st_k = 0

    # generate config tree from queue
    for field in queue:

        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]

        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        # this block will add an extra newlne to any string that is not preceded by whitespace.
        # the idea is that higher level dict objects will be separated by newline character, making easy to read

        # Pattern to match newline followed by a non-whitespace character
        pattern = r"(?<=\n)(\S)"

        # Replacement pattern with newline followed by two newlines
        replacement = r"\n\n\1"
        # Perform the substitution
        # branch_content = re.sub(pattern, replacement, branch_content)

        # print(field)
        # print(styles[st_k])

        # if field=='callbacks':
        #    print('pausing here=')

        # nb you can prob insert newline with lookahead function but too compllicatd to do quickly:
        # https://www.regular-expressions.info/lookaround.html
        # x = re.sub("\n\S", "\n\n", branch_content)

        # [\r\n]

        # rm=re.match('\n',branch_content)
        # (\r\n|\r|\n)

        # branch.add(rich.syntax.Syntax(code=branch_content,lexer= "yaml",indent_guides=True,theme=styles[st_k]))
        branch.add(rich.syntax.Syntax(code=branch_content, lexer="yaml", indent_guides=True, theme="dracula"))

        st_k += 1

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)

    # \nm


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
