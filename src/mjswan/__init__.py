"""mjswan: Browser-based MuJoCo Playground

Interactive MuJoCo simulations with ONNX policies running entirely in the browser.
"""

__version__ = "0.3.0"

from .app import mjswanApp
from .builder import Builder
from .command import (
    Button,
    ButtonConfig,
    CommandGroupConfig,
    CommandInput,
    Slider,
    SliderConfig,
    velocity_command,
)
from .envs.mdp.actions import (
    ActionTermCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from .managers.termination_manager import TerminationTermCfg
from .policy import PolicyConfig, PolicyHandle
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig, SceneHandle
from .splat import SplatConfig, SplatHandle
from .viewer_config import ViewerConfig

__all__ = [
    # Builder and App
    "Builder",
    "mjswanApp",
    # Handles
    "ProjectHandle",
    "SceneHandle",
    "SplatHandle",
    "PolicyHandle",
    # Configs
    "ProjectConfig",
    "SceneConfig",
    "ViewerConfig",
    "SplatConfig",
    "PolicyConfig",
    # MDP config (mjlab-compatible)
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "ActionTermCfg",
    "JointPositionActionCfg",
    "JointEffortActionCfg",
    "TerminationTermCfg",
    # Commands
    "Slider",
    "SliderConfig",
    "Button",
    "ButtonConfig",
    "CommandGroupConfig",
    "CommandInput",
    "velocity_command",
]
